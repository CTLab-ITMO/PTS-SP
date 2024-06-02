import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch
import numpy as np
import cv2

torch.manual_seed(1337)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CustomDataset(Dataset):
    def __init__(
        self,
        data_dict,
        data_dict_values,
        transform=None,
        max_combinations=None,
        split_ratios=(0.7, 0.15, 0.15),
        split="train",
    ):
        """
        :param data_dict: Dictionary of data paths.
        :param data_dict_values: Additional meta information for the data.
        :param transform: Image transformations.
        :param max_combinations: Max number of combinations to generate.
        :param split_ratios: Tuple of train, validation, and test split ratios.
        :param split: 'train', 'val', or 'test' to specify the dataset split.
        """
        self.data_dict = data_dict
        self.data_dict_values = data_dict_values
        self.transform = transform
        self.max_combinations = max_combinations
        self.split_ratios = split_ratios
        self.split = split
        random.seed(10)
        self.samples = self.generate_samples()

    def generate_samples(self):
        samples = []
        for dataset_path in self.data_dict.keys():
            class_index = int(dataset_path.split("/")[3].split("-")[-1])
            class_values = self.data_dict[dataset_path]
            if class_values == []:
                continue
            class_folder = os.path.join(dataset_path, "train", "images")
            label_folder = os.path.join(dataset_path, "train", "labels")
            learning_curve_data = self.data_dict_values[dataset_path]
            if os.path.isdir(class_folder):
                all_images = [
                    os.path.join(class_folder, file)
                    for file in os.listdir(class_folder)
                    if file.endswith((".jpg", ".png"))
                ]
                all_labels = [
                    os.path.join(label_folder, file)
                    for file in os.listdir(class_folder)
                    if file.endswith((".txt"))
                ]
                # Select the appropriate split
                split_images = all_images

                num_images = len(split_images)
                if num_images >= 4:
                    combinations_to_generate = self.calculate_combinations(
                        num_images, 4
                    )
                    combinations_to_generate = (
                        min(combinations_to_generate, self.max_combinations)
                        if self.max_combinations
                        else combinations_to_generate
                    )

                    generated_combinations = set()
                    while len(generated_combinations) < combinations_to_generate:
                        indices = tuple(sorted(random.sample(range(num_images), 4)))
                        if indices not in generated_combinations:
                            generated_combinations.add(indices)
                            combination_images = [split_images[i] for i in indices]
                            combination_labels = [
                                label_folder
                                + "/"
                                + image_path.split("/")[-1].split(".")[0]
                                + ".txt"
                                for image_path in combination_images
                            ]
                            samples.append(
                                (
                                    combination_images,
                                    combination_labels,
                                    class_index,
                                    class_values,
                                    learning_curve_data,
                                )
                            )
        return samples

    @staticmethod
    def split_data(images, ratios):
        random.seed(10)
        random.shuffle(images)
        train_end = int(len(images) * ratios[0])
        val_end = train_end + int(len(images) * ratios[1])
        return images[:train_end], images[train_end:val_end], images[val_end:]

    @staticmethod
    def calculate_combinations(n, r):
        from math import factorial

        return factorial(n) // (factorial(r) * factorial(n - r))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, label_paths, class_index, target_values, learning_curve_data = (
            self.samples[idx]
        )
        images = []
        labels = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        image_width = 800
        image_height = 800
        for label_path in label_paths:
            with open(label_path, "r") as file:
                lines = file.readlines()
                coordinates = []
                for line in lines:
                    if line.strip().startswith(str(class_index)):
                        coords = list(map(float, line.strip().split()[1::]))
                        pixel_coords = []
                        for i in range(0, len(coords), 2):
                            x = int(coords[i] * image_width)
                            y = int(coords[i + 1] * image_height)
                            pixel_coords.append((x, y))
                        coordinates.append(pixel_coords)
            mask_coordinates = coordinates
            mask = np.zeros(
                (image_width, image_height), dtype=np.uint8
            )  # Create a blank mask with the same height and width as the image
            for coords in mask_coordinates:
                # Convert the list of tuples to a numpy array
                points = np.array(coords, dtype=np.int32)
                cv2.fillPoly(
                    mask, [points], 255
                )  # Fill the polygon defined by points with white color (255)
            label_image = Image.fromarray(mask)
            if self.transform:
                label_image = self.transform(label_image)
            labels.append(label_image)
        train_sizes, map_values_dict = learning_curve_data
        map_values = [
            map_values_dict[class_id][0] for class_id in sorted(map_values_dict)
        ]
        labels = torch.stack(labels).repeat(1, 3, 1, 1)
        return (
            torch.stack(images),
            labels,
            torch.tensor(target_values),
            torch.tensor(train_sizes),
            torch.tensor(map_values),
        )
