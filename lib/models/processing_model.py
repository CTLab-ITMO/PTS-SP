import ruamel.yaml
from ultralytics import YOLO
from pathlib import Path
import os
from collections import defaultdict
from typing import List, Dict
import copy
import shutil
import numpy as np
import psutil
import gc
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ctypes
import ctypes.util
import torch


class SizeToMetricModel:
    def __init__(
        self,
        path_to_model: str,
        path_to_yaml: str,
        train_perc: float,
        test_perc: float,
        val_perc: float,
        ttvs_flag=0,
        tp_flag=0,
    ):
        """Инициализация переменных

        Args:
            path_to_model (str): путь до весов yolov8.pt
            path_to_yaml (str): путь до data.yaml файла датасета
            train_perc (float): доля тренировочных данных
            test_perc (float): доля тестовых данных
            val_perc (float): доля валидационных данных
        """
        self.path_to_model = path_to_model
        self.path_to_yaml = path_to_yaml
        self.train_path = Path("train") if os.path.exists(Path("train")) else None
        self.test_path = Path("test") if os.path.exists(Path("test")) else None
        self.val_path = Path("valid") if os.path.exists(Path("valid")) else None
        self.train_perc = train_perc
        self.test_perc = test_perc
        self.val_perc = val_perc

        self.ttvs_flag = ttvs_flag
        self.tp_flag = tp_flag

        self.ram_usage = []

        assert self.train_path != None, "Директория train отсутствует"
        if self.val_path == None:
            os.mkdir("valid", mode=0o777)
            os.mkdir("valid/images", mode=0o777)
            os.mkdir("valid/labels", mode=0o777)
            self.val_path = Path("valid")
        if self.test_path == None:
            os.mkdir("test", mode=0o777)
            os.mkdir("test/images", mode=0o777)
            os.mkdir("test/labels", mode=0o777)
            self.test_path = Path("test")

    def get_ram_usage(
        self,
    ):
        """Gets the current RAM usage of the system.

        Returns:
            float: RAM usage in GB.
        """
        return psutil.virtual_memory().used / 1e9

    def train(self, folder_name: str, iters: int) -> YOLO:
        """Инициализация модели и обучение

        Args:
            folder_name (str): название директории с частью данных
            iters (int): кол-во интераций

        Returns:
            YOLO: экземпляр обученной модели
        """
        # Корректируем data.yaml файл
        yaml = ruamel.yaml.YAML()
        # yaml.preserve_quotes = True
        with open("data.yaml", "r+") as fp:
            data = yaml.load(fp)
            for elem in data:
                if elem == "train":
                    data[elem] = "./" + folder_name + "/train/images"
                elif elem == "val":
                    data[elem] = f"./valid_{folder_name.split('_')[1]}/images"
                elif elem == "test":
                    data[elem] = f"./test_{folder_name.split('_')[1]}/images"
            fp.truncate(0)
            fp.seek(0)
            yaml.dump(data, fp)
        self.ram_usage.append(self.get_ram_usage())
        model = YOLO(self.path_to_model)
        model.train(
            # Random Seed parameters
            deterministic=True,
            seed=43,
            data=self.path_to_yaml,
            pretrained=True,
            exist_ok=True,
            epochs=iters,
            optimizer="SGD",
            device=0,
            plots=False,
        )

        return model

    def retrain(
        self,
        folder_name: str,
        iters: int,
    ) -> YOLO:
        """Обучение модели на всех предыдущих частях данных для профилакти просадки метрики

        Args:
            folder_name (str): название директории с частью данных
            iters (int): кол-во интераций

        Returns:
            YOLO: экземпляр обученной модели
        """
        os.mkdir("retrain", mode=0o777)
        os.mkdir("retrain/train", mode=0o777)
        os.mkdir("retrain/train/images", mode=0o777)
        os.mkdir("retrain/train/labels", mode=0o777)

        # собираем список всех кусков данных до нашего folder_name
        folder_num = np.arange(1, int(folder_name.split("_")[-1]) + 1)
        cls = folder_name.split("_")[1]
        source_pathes = [f"temp_{cls}_{i}" for i in folder_num]

        # копируем все собранные куски данных в папку retrain
        for path in source_pathes:
            # собираем все файлы
            all_images_train = os.listdir(path + "/train/images")
            all_labels_train = os.listdir(path + "/train/labels")

            for image in all_images_train:
                shutil.copyfile(
                    path + "/train/images/" + image, "retrain/train/images/" + image
                )
            for label in all_labels_train:
                shutil.copyfile(
                    path + "/train/labels/" + label, "retrain/train/labels/" + label
                )

        # Корректируем data.yaml файл
        yaml = ruamel.yaml.YAML()
        with open("data.yaml", "r+") as fp:
            data = yaml.load(fp)
            for elem in data:
                if elem == "train":
                    data[elem] = "retrain/train/images"
                elif elem == "val":
                    data[elem] = f"./valid_{folder_name.split('_')[1]}/images"
                elif elem == "test":
                    data[elem] = f"./test_{folder_name.split('_')[1]}/images"
            fp.truncate(0)
            fp.seek(0)
            yaml.dump(data, fp)

        self.ram_usage.append(self.get_ram_usage())
        # load a pretrained model (recommended for training)
        model = YOLO(self.path_to_model)
        model.train(  # Random Seed parameters
            deterministic=True,
            seed=43,
            data=self.path_to_yaml,
            pretrained=True,
            exist_ok=True,
            epochs=iters,
            optimizer="SGD",
            device=0,
            plots=False,
        )

        return model

    def test(self, folder_name: str, model: YOLO):
        """Тестирование модели

        Args:
            model (YOLO): экземпляр обученной модели

        Returns:
            _type_: _description_
        """
        # Корректируем data.yaml файл
        yaml = ruamel.yaml.YAML()
        with open("data.yaml", "r+") as fp:
            data = yaml.load(fp)
            for elem in data:
                if elem == "train":
                    data[elem] = "retrain/train/images"
                elif elem == "val":
                    data[elem] = f"./valid_{folder_name.split('_')[1]}/images"
                elif elem == "test":
                    data[elem] = f"./test_{folder_name.split('_')[1]}/images"
            fp.truncate(0)
            fp.seek(0)
            yaml.dump(data, fp)

        metrics = model.val(data=self.path_to_yaml, split="test")
        return metrics

    def train_test_val_split(self, keep_perc: float, learn_cls: str = None):
        """Разделение изначального датасета на заданные доли train/test/val. Удаление 1-keep_perc доли данных

        Args:
            keep_perc (float): доля данных, которую нужно оставить
        """
        # создаем директории для объединения всех файлов
        os.mkdir("temp", mode=0o777)
        os.mkdir("temp/images", mode=0o777)
        os.mkdir("temp/labels", mode=0o777)

        source_pathes = (
            self.train_path / "images",
            self.train_path / "labels",
            self.test_path / "images",
            self.test_path / "labels",
            self.val_path / "images",
            self.val_path / "labels",
        )
        destination = Path("temp")

        for path in source_pathes:
            # собираем все файлы
            allfiles = os.listdir(path)
            # итерируем по всем файлам, чтобы переместить их в папку назначения
            sub_folder = path.name  # images or labels
            for f in allfiles:
                src_path = os.path.join(path, f)
                dst_path = os.path.join(destination / sub_folder, f)
                os.rename(src_path, dst_path)
        total_num = len(allfiles)
        # Создаем словарь, где ключ - метка класса, значение - множество путей к label файлам данного класса
        classes = defaultdict(set)
        empty_count = 0
        for txt_path in Path("temp/labels").glob("*.txt"):
            with txt_path.open() as f:
                text = f.read().strip()
                # Каждому объекту на изображении соотвествует одна строка, где первое значение - метка класса,
                # а остальное - координаты сегментации
                for obj in text.split("\n"):
                    if len(obj) > 0:
                        classes[obj.split()[0]].add(txt_path)
                    else:
                        print(f"Пустой файл: {txt_path}")
                        print(text)
                        empty_count += 1
        print(f"Кол-во пустых файлов - {empty_count}")
        # Оставляем указанный процент данных
        if learn_cls == None:
            for cls, pathes in zip(list(classes.keys()), list(classes.values())):
                num_files = len(pathes)
                num_to_del = num_files * (1 - keep_perc)
                for i, file_path in enumerate(pathes.copy()):
                    if i + 1 >= num_to_del:
                        break
                    f = file_path.name.split(".")[:-1]
                    f.append("png")
                    try:
                        Path("temp", "images", ".".join(f)).unlink()
                        file_path.unlink()
                    except OSError as e:
                        # Файл не найден (скорее всего имеет несколько классов и был уже перемещен)
                        pass
                    classes[cls].remove(file_path)
        else:
            cls = learn_cls
            pathes = classes[cls]
            num_files = len(pathes)
            num_to_del = num_files * (1 - keep_perc)
            for i, file_path in enumerate(pathes.copy()):
                if i + 1 >= num_to_del:
                    break
                f = file_path.name.split(".")[:-1]
                f.append("png")
                try:
                    Path("temp", "images", ".".join(f)).unlink()
                    file_path.unlink()
                except OSError as e:
                    # Файл не найден (скорее всего имеет несколько классов и был уже перемещен)
                    pass
                classes[cls].remove(file_path)
        if learn_cls == None:
            for cls in classes.keys():
                #             shutil.copyfile("data.yaml", f"data_{cls}.yaml")
                #              # Корректируем data.yaml файл
                #             yaml = ruamel.yaml.YAML()
                #             with open(f'data_{cls}.yaml', 'r+') as fp:
                #                 data = yaml.load(fp)
                #                 data['names'] = [data['names'][int(cls)]]
                #                 data['nc'] = 1
                #                 fp.truncate(0)
                #                 fp.seek(0)
                #                 yaml.dump(data, fp)
                os.mkdir(f"valid_{cls}", mode=0o777)
                os.mkdir(f"test_{cls}", mode=0o777)

                os.mkdir(f"valid_{cls}/labels", mode=0o777)
                os.mkdir(f"valid_{cls}/images", mode=0o777)
                os.mkdir(f"test_{cls}/labels", mode=0o777)
                os.mkdir(f"test_{cls}/images", mode=0o777)
        else:
            cls = learn_cls
            os.mkdir(f"valid_{cls}", mode=0o777)
            os.mkdir(f"test_{cls}", mode=0o777)
            os.mkdir(f"valid_{cls}/labels", mode=0o777)
            os.mkdir(f"valid_{cls}/images", mode=0o777)
            os.mkdir(f"test_{cls}/labels", mode=0o777)
            os.mkdir(f"test_{cls}/images", mode=0o777)
        # Распределяем данные по директориям train, test, val с учетом указанных пропорций
        class_copy = copy.deepcopy(classes)
        if learn_cls == None:
            for cls, pathes in zip(list(classes.keys()), list(classes.values())):
                num_files = len(class_copy[cls])
                num_to_mv_train = int(num_files * self.train_perc)
                num_to_mv_test = int(num_files * self.test_perc)
                num_to_mv_val = int(num_files * self.val_perc)
                # print(num_files, num_to_mv, len(pathes))
                temp_dict_name = "train"
                for i, file_path in enumerate(pathes.copy()):
                    if (
                        i + 1 > num_to_mv_train
                        and i + 1 < num_to_mv_val + num_to_mv_train
                    ):
                        temp_dict_name = f"valid_{cls}"
                    elif i + 1 > num_to_mv_val + num_to_mv_train:
                        temp_dict_name = f"test_{cls}"
                    f = file_path.name.split(".")[:-1]
                    f.append("png")
                    shutil.copyfile(
                        Path("temp", "images", ".".join(f)),
                        Path(temp_dict_name, "images").joinpath(
                            Path("train", "images", ".".join(f)).name
                        ),
                    )
                    shutil.copyfile(
                        file_path,
                        Path(temp_dict_name, "labels").joinpath(file_path.name),
                    )
                    if temp_dict_name != "train":
                        # remove another classes in label file
                        orig_lines = [
                            line.strip()
                            for line in open(
                                Path(temp_dict_name, "labels").joinpath(file_path.name)
                            )
                        ]
                        new_lines = [l for l in orig_lines if l.split()[0] == str(cls)]

                        with open(
                            Path(temp_dict_name, "labels").joinpath(file_path.name), "w"
                        ) as fp:
                            print(*new_lines, sep="\n", file=fp)
                    classes[cls].remove(file_path)
                dir_path = f"valid_{cls}/images"
                print(
                    dir_path,
                    len(
                        [
                            entry
                            for entry in os.listdir(dir_path)
                            if os.path.isfile(os.path.join(dir_path, entry))
                        ]
                    ),
                )
                dir_path = f"test_{cls}/images"
                print(
                    dir_path,
                    len(
                        [
                            entry
                            for entry in os.listdir(dir_path)
                            if os.path.isfile(os.path.join(dir_path, entry))
                        ]
                    ),
                )
        else:
            cls = learn_cls
            num_files = len(class_copy[cls])
            num_to_mv_train = int(num_files * self.train_perc)
            num_to_mv_test = int(num_files * self.test_perc)
            num_to_mv_val = int(num_files * self.val_perc)
            # print(num_files, num_to_mv, len(pathes))
            temp_dict_name = "train"
            for i, file_path in enumerate(pathes.copy()):
                if i + 1 > num_to_mv_train and i + 1 < num_to_mv_val + num_to_mv_train:
                    temp_dict_name = f"valid_{cls}"
                elif i + 1 > num_to_mv_val + num_to_mv_train:
                    temp_dict_name = f"test_{cls}"
                f = file_path.name.split(".")[:-1]
                f.append("png")
                shutil.copyfile(
                    Path("temp", "images", ".".join(f)),
                    Path(temp_dict_name, "images").joinpath(
                        Path("train", "images", ".".join(f)).name
                    ),
                )
                shutil.copyfile(
                    file_path, Path(temp_dict_name, "labels").joinpath(file_path.name)
                )
                if temp_dict_name != "train":
                    # remove another classes in label file
                    orig_lines = [
                        line.strip()
                        for line in open(
                            Path(temp_dict_name, "labels").joinpath(file_path.name)
                        )
                    ]
                    new_lines = [l for l in orig_lines if l.split()[0] == str(cls)]

                    with open(
                        Path(temp_dict_name, "labels").joinpath(file_path.name), "w"
                    ) as fp:
                        print(*new_lines, sep="\n", file=fp)
                classes[cls].remove(file_path)
            dir_path = f"valid_{cls}/images"
            print(
                dir_path,
                len(
                    [
                        entry
                        for entry in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, entry))
                    ]
                ),
            )
            dir_path = f"test_{cls}/images"
            print(
                dir_path,
                len(
                    [
                        entry
                        for entry in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, entry))
                    ]
                ),
            )
        dir_path = f"train/labels"
        print(
            dir_path,
            len(
                [
                    entry
                    for entry in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, entry))
                ]
            ),
            "\n",
        )

        shutil.rmtree("temp")

    def __train_set_of(self, n):
        x = []
        temp = 2
        while temp < n:
            x.append(temp)
            if temp < 10:
                temp += 2
            elif temp >= 10 and temp < 30:
                temp += 3
            elif temp >= 30 and temp < 100:
                temp += 5
            elif temp >= 100 and temp < 200:
                temp += 10
            elif temp >= 200 and temp < 300:
                temp += 50
            elif temp >= 300 and temp < 500:
                temp += 100
            elif temp >= 500:
                temp += 500
        if x[-1] != n:
            x.append(n)

        return x

    def take_piece(
        self, piece_perc: float = None, fib_flag: bool = False, learn_cls: str = None
    ) -> (dict, dict):
        """Разделение исходного датасета на части равные доле piece_perc по директориям temp_{n}, n = {1..1/piece_perc}

        Args:
            piece_perc (float): доля части данных, на которые нужно поделить датасет
        """

        source_pathes = (self.train_path / "images", self.train_path / "labels")

        # Создаем словарь, где ключ - метка класса, значение - множество путей к label файлам данного класса
        classes = defaultdict(set)
        empty_count = 0
        for txt_path in Path("train/labels").glob("*.txt"):
            with txt_path.open() as f:
                text = f.read().strip()
                # Каждому объекту на изображении соотвествует одна строка, где первое значение - метка класса,
                # а остальное - координаты сегментации
                for obj in text.split("\n"):
                    if len(obj) > 0:
                        classes[obj.split()[0]].add(txt_path)
                    else:
                        print(f"Пустой файл: {txt_path}")
                        empty_count += 1

        print(f"Кол-во пустых файлов - {empty_count}")
        d_c = {key: len(value) for key, value in zip(classes.keys(), classes.values())}
        print(
            *[
                f"Класс {key} содержит {value} объекта(-ов)\n"
                for key, value in zip(d_c.keys(), d_c.values())
            ]
        )

        cls_tl_dict = defaultdict(
            list
        )  # key - class, value - result of __train_set_of(n)
        cls_fif_dict = defaultdict(int)  # key - class, value - num folders
        # Разделить сначала по классам, а потом внутри класса разделить по __train_set_of
        if learn_cls == None:
            for cls in classes.keys():
                print(f"Класс {cls}")
                total_num = len(classes[cls])
                print(f"\tКол-во train класса {cls}: {total_num}")
                if fib_flag == True:
                    n = total_num
                    train_list = self.__train_set_of(n)
                    files_in_folder = []
                    for i in range(len(train_list)):
                        if i == 0:
                            files_in_folder.append(train_list[i])
                            continue
                        files_in_folder.append(train_list[i] - train_list[i - 1])
                    print(
                        f"\tКоличество данных (train) на каждой итерации класса {cls}: {train_list}"
                    )
                    cls_tl_dict[cls] = train_list
                    cls_fif_dict[cls] = len(files_in_folder)

                if fib_flag == True:
                    self.num_folders = len(files_in_folder)
                    print(f"\tКол-во директорий для класса {cls}: {self.num_folders} ")
                else:
                    self.num_folders = 1 / piece_perc
                for folder in range(int(self.num_folders)):
                    os.mkdir(f"temp_{cls}_{folder+1}", mode=0o777)

                    os.mkdir(f"temp_{cls}_{folder+1}/train", mode=0o777)

                    os.mkdir(f"temp_{cls}_{folder+1}/train/labels", mode=0o777)
                    os.mkdir(f"temp_{cls}_{folder+1}/train/images", mode=0o777)

                # Распределяем данные по директориям
                class_copy = copy.deepcopy(classes)
                for f_i, folder in enumerate(range(int(self.num_folders))):
                    folder_name = f"temp_{cls}_{folder+1}"
                    num_to_mv_train = (
                        int(num_files * piece_perc)
                        if fib_flag == False
                        else files_in_folder[f_i]
                    )
                    print(
                        f"\tnum_to_mv_train {num_to_mv_train}, folder {folder}, cls {cls}"
                    )
                    # print(num_files, num_to_mv, len(pathes))
                    temp_dict_name = "train"
                    for i, file_path in enumerate(classes[cls].copy()):
                        if i + 1 > num_to_mv_train:
                            break
                        f = file_path.name.split(".")[:-1]
                        f.append("png")
                        shutil.copyfile(
                            Path("train", "images", ".".join(f)),
                            Path(folder_name, temp_dict_name, "images").joinpath(
                                Path("train", "images", ".".join(f)).name
                            ),
                        )
                        shutil.copyfile(
                            file_path,
                            Path(folder_name, temp_dict_name, "labels").joinpath(
                                file_path.name
                            ),
                        )
                        # remove another classes in label file
                        orig_lines = [
                            line.strip()
                            for line in open(
                                Path(folder_name, temp_dict_name, "labels").joinpath(
                                    file_path.name
                                )
                            )
                        ]
                        new_lines = [l for l in orig_lines if l.split()[0] == str(cls)]

                        with open(
                            Path(folder_name, temp_dict_name, "labels").joinpath(
                                file_path.name
                            ),
                            "w",
                        ) as fp:
                            print(*new_lines, sep="\n", file=fp)
                        classes[cls].remove(file_path)
                for folder in range(int(self.num_folders)):
                    dir_path = f"temp_{cls}_{folder+1}/train/labels"
                    print(
                        dir_path,
                        len(
                            [
                                entry
                                for entry in os.listdir(dir_path)
                                if os.path.isfile(os.path.join(dir_path, entry))
                            ]
                        ),
                    )
                    dir_path = f"temp_{cls}_{folder+1}/train/images"
                    print(
                        dir_path,
                        len(
                            [
                                entry
                                for entry in os.listdir(dir_path)
                                if os.path.isfile(os.path.join(dir_path, entry))
                            ]
                        ),
                        "\n",
                    )
        else:
            cls = learn_cls
            print(f"Класс {cls}")
            total_num = len(classes[cls])
            print(f"\tКол-во train класса {cls}: {total_num}")
            if fib_flag == True:
                n = total_num
                train_list = self.__train_set_of(n)
                files_in_folder = []
                for i in range(len(train_list)):
                    if i == 0:
                        files_in_folder.append(train_list[i])
                        continue
                    files_in_folder.append(train_list[i] - train_list[i - 1])
                print(
                    f"\tКоличество данных (train) на каждой итерации класса {cls}: {train_list}"
                )
                cls_tl_dict[cls] = train_list
                cls_fif_dict[cls] = len(files_in_folder)

            if fib_flag == True:
                self.num_folders = len(files_in_folder)
                print(f"\tКол-во директорий для класса {cls}: {self.num_folders} ")
            else:
                self.num_folders = 1 / piece_perc
            for folder in range(int(self.num_folders)):
                os.mkdir(f"temp_{cls}_{folder+1}", mode=0o777)

                os.mkdir(f"temp_{cls}_{folder+1}/train", mode=0o777)

                os.mkdir(f"temp_{cls}_{folder+1}/train/labels", mode=0o777)
                os.mkdir(f"temp_{cls}_{folder+1}/train/images", mode=0o777)

            # Распределяем данные по директориям
            class_copy = copy.deepcopy(classes)
            for f_i, folder in enumerate(range(int(self.num_folders))):
                folder_name = f"temp_{cls}_{folder+1}"
                num_to_mv_train = (
                    int(num_files * piece_perc)
                    if fib_flag == False
                    else files_in_folder[f_i]
                )
                print(
                    f"\tnum_to_mv_train {num_to_mv_train}, folder {folder}, cls {cls}"
                )
                # print(num_files, num_to_mv, len(pathes))
                temp_dict_name = "train"
                for i, file_path in enumerate(classes[cls].copy()):
                    if i + 1 > num_to_mv_train:
                        break
                    f = file_path.name.split(".")[:-1]
                    f.append("png")
                    shutil.copyfile(
                        Path("train", "images", ".".join(f)),
                        Path(folder_name, temp_dict_name, "images").joinpath(
                            Path("train", "images", ".".join(f)).name
                        ),
                    )
                    shutil.copyfile(
                        file_path,
                        Path(folder_name, temp_dict_name, "labels").joinpath(
                            file_path.name
                        ),
                    )
                    # remove another classes in label file
                    orig_lines = [
                        line.strip()
                        for line in open(
                            Path(folder_name, temp_dict_name, "labels").joinpath(
                                file_path.name
                            )
                        )
                    ]
                    new_lines = [l for l in orig_lines if l.split()[0] == str(cls)]

                    with open(
                        Path(folder_name, temp_dict_name, "labels").joinpath(
                            file_path.name
                        ),
                        "w",
                    ) as fp:
                        print(*new_lines, sep="\n", file=fp)
                    classes[cls].remove(file_path)
            for folder in range(int(self.num_folders)):
                dir_path = f"temp_{cls}_{folder+1}/train/labels"
                print(
                    dir_path,
                    len(
                        [
                            entry
                            for entry in os.listdir(dir_path)
                            if os.path.isfile(os.path.join(dir_path, entry))
                        ]
                    ),
                )
                dir_path = f"temp_{cls}_{folder+1}/train/images"
                print(
                    dir_path,
                    len(
                        [
                            entry
                            for entry in os.listdir(dir_path)
                            if os.path.isfile(os.path.join(dir_path, entry))
                        ]
                    ),
                    "\n",
                )
        shutil.rmtree("train")
        return cls_tl_dict, cls_fif_dict

    def plot_result(self, result_dict: dict, color_dict="green", fib_list: list = None):
        """Функция для отрисовки графиков зависимости метрик от размера обучающей выборки

        Args:
            result_dict (dict): словарь с метриками {доля данных: массив метрик}
            color_dict (dict): словарь с индикаторами повторного обучения
        """
        metrics_names = ("mAP", "mAP50", "mAP75")
        colors = list(color_dict.values()) if color_dict != "green" else color_dict
        if fib_list == None:
            for i, metric in enumerate(metrics_names):
                plt.scatter(
                    list(result_dict.keys()),
                    [j[i] for j in result_dict.values()],
                    color=colors,
                    zorder=1,
                )
                plt.plot(
                    list(result_dict.keys()),
                    [j[i] for j in result_dict.values()],
                    linestyle="-",
                )
                plt.ylabel(metric)
                plt.xlabel("Num files (train)")
                plt.xticks(ticks=list(result_dict.keys()))
                plt.show()
        else:
            for i, metric in enumerate(metrics_names):
                plt.scatter(
                    fib_list,
                    [j[i] for j in result_dict.values()],
                    color=colors,
                    zorder=1,
                )
                plt.plot(fib_list, [j[i] for j in result_dict.values()], linestyle="-")
                plt.ylabel(metric)
                plt.xlabel("Num files (train)")
                plt.xticks(ticks=list(fib_list))
                plt.show()

    def plot_ram_usage(
        self,
    ):
        """Функция для отрисовки использования RAM в процессе обучения"""
        plt.plot(self.ram_usage, marker="o", linestyle="-", color="b")
        plt.xlabel("Iteration")
        plt.ylabel("RAM Usage (GB)")
        plt.title("RAM Usage During Training")
        plt.show()

    def __early_stopping(
        self, result_dict: dict, prev_num: int, threshold: float
    ) -> bool:
        keys_sorted = sorted(result_dict.keys())
        prev_results = np.array(
            [result_dict[i][0] for i in keys_sorted[-prev_num - 1 : -1]]
        )
        res = result_dict[keys_sorted[-1]][0] - prev_results
        print(
            f"Progress for {prev_num} last iterations with {threshold} threshold = {np.sum(res)}"
        )
        if np.sum(res) < threshold:
            return True
        else:
            return False

    def increm_learning(
        self,
        keep_perc: float,
        iters: int,
        piece_perc: float = None,
        fib_flag: bool = None,
        prev_num: int = 3,
        threshold: float = 0.01,
    ) -> (list, list):
        os.environ["WANDB_DISABLED"] = "true"
        # делим датасет на тренировочную/тестовую/валидационную выборку
        if self.ttvs_flag == 0:
            self.train_test_val_split(keep_perc)
            self.ttvs = 1
        if self.tp_flag == 0:
            cls_tl_dict, cls_fif_dict = self.take_piece(piece_perc, fib_flag)
            self.tp = 1
        else:
            assert fib_flag == False
            self.num_folders = 1 / piece_perc
        # путь к изначальным весам yolov8 для повторного обучения при понижении метрики
        native_path_to_model = self.path_to_model
        # словарь с метриками {доля данных: массив метрик}
        # Инкрементальное обучение
        for cls in cls_tl_dict.keys():
            #             self.path_to_yaml = "/".join(self.path_to_yaml.split('/')[0:-1]) + f"/data_{cls}.yaml"
            print(self.path_to_yaml)
            result_dict = defaultdict(list)
            # словарь с индикаторами повторного обучения
            color_dict = defaultdict(str)
            # переменная для отслеживания максимального map в целях профилактики просадок метрики
            max_map = 0
            self.path_to_model = native_path_to_model
            for folder in range(cls_fif_dict[cls]):
                if folder > prev_num:
                    if self.__early_stopping(result_dict, prev_num, threshold):
                        break
                libc = ctypes.CDLL(ctypes.util.find_library("c"))
                libc.malloc_trim(ctypes.c_int(0))

                torch.set_num_threads(1)
                folder_name = f"temp_{cls}_{folder+1}"
                # дообучаем модель
                model = self.train(folder_name, iters)
                # тестируем модель
                metrics = self.test(folder_name, model)
                # проверяем, что метрика улучшается
                if metrics.seg.map > max_map:
                    max_map = metrics.seg.map
                    # заносим метрики в словарь
                    result_dict[folder].append(metrics.seg.map)
                    result_dict[folder].append(metrics.seg.map50)
                    result_dict[folder].append(metrics.seg.map75)
                    color_dict[folder] = "green"
                    #                     clear_output(wait=True)
                    del model
                    del metrics
                    gc.collect()
                    self.path_to_model = "./runs/segment/train/weights/last.pt"
                else:
                    self.path_to_model = native_path_to_model
                    # дообучаем модель
                    model = self.retrain(folder_name, iters)
                    # тестируем модель
                    metrics = self.test(folder_name, model)
                    if metrics.seg.map > max_map:
                        max_map = metrics.seg.map
                    # заносим метрики в словарь
                    result_dict[folder].append(metrics.seg.map)
                    result_dict[folder].append(metrics.seg.map50)
                    result_dict[folder].append(metrics.seg.map75)
                    color_dict[folder] = "blue"
                    #                     clear_output(wait=True)
                    del model
                    del metrics
                    gc.collect()
                    self.path_to_model = "./runs/segment/train/weights/last.pt"
                    shutil.rmtree("retrain")

            print(
                f"Итоговый результат (инкрементальное обучение) для класса {cls}: \n {result_dict}"
            )
            print(f"Количество данных (train) для класса {cls}: {cls_tl_dict[cls]}")
            (
                self.plot_result(result_dict, color_dict)
                if fib_flag == False
                else self.plot_result(
                    result_dict, color_dict, cls_tl_dict[cls][: len(result_dict.keys())]
                )
            )

    def increm_learning_one_class(
        self,
        learn_cls: str,
        keep_perc: float,
        iters: int,
        piece_perc: float = None,
        fib_flag: bool = None,
        prev_num: int = 3,
        threshold: float = 0.01,
    ) -> (list, list):
        os.environ["WANDB_DISABLED"] = "true"
        # делим датасет на тренировочную/тестовую/валидационную выборку
        if self.ttvs_flag == 0:
            self.train_test_val_split(keep_perc, learn_cls)
            self.ttvs = 1
        if self.tp_flag == 0:
            cls_tl_dict, cls_fif_dict = self.take_piece(piece_perc, fib_flag, learn_cls)
            self.tp = 1
        else:
            assert fib_flag == False
            self.num_folders = 1 / piece_perc
        # путь к изначальным весам yolov8 для повторного обучения при понижении метрики
        native_path_to_model = self.path_to_model
        # словарь с метриками {доля данных: массив метрик}
        # Инкрементальное обучение
        cls = learn_cls
        #         self.path_to_yaml = "/".join(self.path_to_yaml.split('/')[0:-1]) + f"/data_{cls}.yaml"
        print(self.path_to_yaml)
        print(cls_fif_dict, cls_tl_dict)
        result_dict = defaultdict(list)
        # словарь с индикаторами повторного обучения
        color_dict = defaultdict(str)
        # переменная для отслеживания максимального map в целях профилактики просадок метрики
        max_map = 0
        self.path_to_model = native_path_to_model
        for folder in range(cls_fif_dict[cls]):
            if folder > prev_num:
                if self.__early_stopping(result_dict, prev_num, threshold):
                    print("EARLY STOPPING")
                    break
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.malloc_trim(ctypes.c_int(0))

            torch.set_num_threads(1)
            folder_name = f"temp_{cls}_{folder+1}"
            # дообучаем модель
            model = self.train(folder_name, iters)
            # тестируем модель
            metrics = self.test(folder_name, model)
            # проверяем, что метрика улучшается
            if metrics.seg.map > max_map:
                max_map = metrics.seg.map
                # заносим метрики в словарь
                result_dict[folder].append(metrics.seg.map)
                result_dict[folder].append(metrics.seg.map50)
                result_dict[folder].append(metrics.seg.map75)
                color_dict[folder] = "green"
                #                     clear_output(wait=True)
                del model
                del metrics
                gc.collect()
                self.path_to_model = "./runs/segment/train/weights/last.pt"
            else:
                self.path_to_model = native_path_to_model
                # дообучаем модель
                model = self.retrain(folder_name, iters)
                # тестируем модель
                metrics = self.test(folder_name, model)
                if metrics.seg.map > max_map:
                    max_map = metrics.seg.map
                # заносим метрики в словарь
                result_dict[folder].append(metrics.seg.map)
                result_dict[folder].append(metrics.seg.map50)
                result_dict[folder].append(metrics.seg.map75)
                color_dict[folder] = "blue"
                #                     clear_output(wait=True)
                del model
                del metrics
                gc.collect()
                self.path_to_model = "./runs/segment/train/weights/last.pt"
                shutil.rmtree("retrain")

        print(
            f"Итоговый результат (инкрементальное обучение) для класса {cls}: \n {result_dict}"
        )
        print(f"Количество данных (train) для класса {cls}: {cls_tl_dict[cls]}")
        (
            self.plot_result(result_dict, color_dict)
            if fib_flag == False
            else self.plot_result(
                result_dict, color_dict, cls_tl_dict[cls][: len(result_dict.keys())]
            )
        )
        return result_dict, cls_tl_dict[cls]

    def base_learning(self, keep_perc: float, piece_perc: float, iters: int):
        os.environ["WANDB_DISABLED"] = "true"
        # делим датасет на тренировочную/тестовую/валидационную выборку
        if self.ttvs_flag == 0:
            self.train_test_val_split(keep_perc)
            self.ttvs = 1
        if self.tp_flag == 0:
            self.take_piece(piece_perc)
            self.tp_flag = 1
        else:
            self.num_folders = 1 / piece_perc
        # путь к изначальным весам yolov8 для повторного обучения при понижении метрики
        native_path_to_model = self.path_to_model
        # словарь с метриками {доля данных: массив метрик}
        result_dict = {k_p: [] for k_p in range(int(self.num_folders))}

        for folder in range(int(self.num_folders)):
            folder_name = f"temp_{folder+1}"
            # дообучаем модель
            model = self.retrain(folder_name, iters)
            # тестируем модель
            metrics = self.test(model)
            # заносим метрики в словарь
            result_dict[folder].append(metrics.seg.map)
            result_dict[folder].append(metrics.seg.map50)
            result_dict[folder].append(metrics.seg.map75)
            clear_output(wait=True)
            del model
            del metrics
            gc.collect()
            shutil.rmtree("retrain")

        print(f"Итоговый результат (базовое обучение): \n {result_dict}")
        self.plot_result(result_dict)

        return result_dict
