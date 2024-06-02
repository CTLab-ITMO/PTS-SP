from roboflow import Roboflow
from pathlib import Path
import os
from typing import List, Dict
import numpy as np
import gc
from IPython.display import clear_output
import ctypes
import ctypes.util
import torch
import random
import json
from ultralytics.utils import SETTINGS
from functools import lru_cache
from torchvision import models, transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def setup_environment(seed: int = 43) -> None:
    """
    Set up the environment by disabling WANDB, setting random seeds, and ensuring deterministic behavior.

    Args:
        seed (int): The seed value for reproducibility. Default is 43.
    """
    SETTINGS["wandb"] = False
    os.environ["WANDB_DISABLED"] = "true"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # trying to optimize RAM usage
    gc.enable()

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.malloc_trim(ctypes.c_int(0))

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def download_dataset(
    api_key: str, workspace: str, project_name: str, version: int, data_type: str
) -> Path:
    """
    Download dataset using Roboflow API.

    Args:
        api_key (str): The API key for Roboflow.
        workspace (str): The workspace name.
        project_name (str): The project name.
        version (int): The version of the dataset.
        data_type (str): The type of data to download.

    Returns:
        Path: The path to the downloaded dataset.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download(data_type)
    return dataset


def save_processed_data(class_num, result_dict, cls_tl_dict):
    data = {
        "class_num": class_num,
        "result_dict": result_dict,
        "cls_tl_dict": cls_tl_dict,
    }
    filename = "processed_result.json"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)
        existing_data.append(data)
        with open(filename, "w") as file:
            json.dump(existing_data, file, indent=4)
    else:
        with open(filename, "w") as file:
            json.dump([data], file, indent=4)


def load_processed_data():
    filename = "processed_result.json"
    if not os.path.exists(filename):
        return {}

    with open(filename, "r") as file:
        data = json.load(file)

    execution_data = {}
    for i, entry in enumerate(data):
        class_num_str = str(entry["class_num"])
        execution_data[str(i)] = [entry["cls_tl_dict"], entry["result_dict"]]

    return execution_data


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def log4pl(x, A, B, C, D):
    return ((A - D) / (1.0 + ((x / C) ** B))) + D


@lru_cache
def incept_init(dump: int):
    incept = models.inception_v3(pretrained=True).to(device)

    # Remove the final fully connected layer to get embeddings from the penultimate layer
    incept.fc = torch.nn.Identity()

    # Ensure the model is in evaluation mode
    incept.eval()
    return incept


def get_incept_image_embedding(image_tensor):
    incept = incept_init(1)
    # Resize and normalize the image tensor to fit InceptionV3 input requirements
    preprocess = transforms.Compose(
        [
            transforms.Resize(299),  # Resize the image to 299x299
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = preprocess(image_tensor)
    # No need to track gradients for this operation
    with torch.no_grad():
        # Get the embeddings for the image
        embeddings = incept(image_tensor)

    return embeddings


def evaluate(model, loader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, targets, train_sizes, empirical_maps in tqdm(loader):
            inputs, labels, targets = (
                inputs.to(device),
                labels.to(device),
                targets.to(device),
            )
            empirical_maps = empirical_maps.to(device)  # Empirical mAP values

            outputs = model(
                torch.cat(
                    (get_image_embedding(inputs[0]), get_image_embedding(labels[0])),
                    dim=1,
                )
            )  # Predicted parameters
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    # Calculate SMAPE for each parameter
    smapes = []
    for i in range(
        all_targets.shape[1]
    ):  # Assuming the second dimension is the parameter dimension
        smape_value = smape(all_targets[:, i], all_predictions[:, i])
        smapes.append(smape_value)
    #     avg_smape_sinusoid = total_smape_sinusoid / num_samples
    return smapes


def split_and_process_data(split_ratio=0.5):
    with open("approx_result.json", "r") as file:
        execution_data = json.load(file)

    data_dict_train = {}
    data_dict_test = {}
    data_dict_values_train = {}
    data_dict_values_test = {}

    for class_num_str, data in execution_data.items():
        # Split the data into train and test based on the split ratio
        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]

        # Add train data to data_dict_train
        data_dict_train[class_num_str] = train_data[:4]  # A, B, C, D

        # Add test data to data_dict_test
        data_dict_test[class_num_str] = test_data[:4]  # A, B, C, D

        # Add train data to data_dict_values_train
        data_dict_values_train[class_num_str] = (
            train_data[4:],
            train_data[4 + split_index // 2 :],
        )

        # Add test data to data_dict_values_test
        data_dict_values_test[class_num_str] = (
            test_data[4:],
            test_data[4 + split_index // 2 :],
        )

    return (
        data_dict_train,
        data_dict_test,
        data_dict_values_train,
        data_dict_values_test,
    )
