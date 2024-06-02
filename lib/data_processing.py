from roboflow import Roboflow
import argparse
import wget
import os
import matplotlib.pyplot as plt
import locale
from IPython.display import clear_output
import ctypes.util
from ultralytics.utils import SETTINGS

from .utils import setup_environment, download_dataset, save_processed_data
from models.processing_model import SizeToMetricModel
from .configs.processing_config import RoboflowConfig, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for training and testing model"
    )

    # Roboflow parameters (excluding API key)
    parser.add_argument(
        "--workspace_rf", type=str, required=True, help="Workspace name for Roboflow"
    )
    parser.add_argument(
        "--project_name_rf", type=str, required=True, help="Project name for Roboflow"
    )
    parser.add_argument(
        "--version_rf", type=str, required=True, help="Version of the project"
    )
    parser.add_argument(
        "--data_type_rf", type=str, required=True, help="Data type of the project"
    )

    # Model parameters
    parser.add_argument(
        "--train_perc", type=float, default=0.8, help="Training data percentage"
    )
    parser.add_argument(
        "--test_perc", type=float, default=0.1, help="Testing data percentage"
    )
    parser.add_argument(
        "--val_perc", type=float, default=0.1, help="Validation data percentage"
    )
    parser.add_argument(
        "--keep_perc", type=float, default=1.0, help="Keep percentage of the data"
    )
    parser.add_argument(
        "--piece_perc", type=float, default=0.05, help="Piece percentage of the data"
    )
    parser.add_argument("--iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--fib_flag", type=bool, default=True, help="Fibonacci flag")
    parser.add_argument("--prev_num", type=int, default=50, help="Previous number")
    parser.add_argument(
        "--threshold", type=float, default=0.001, help="Threshold value"
    )
    parser.add_argument("--class_num", type=int, required=True, help="Class number")

    args = parser.parse_args()

    # Extract API key from environment variable
    api_key_rf = os.getenv("API_KEY_RF")
    if not api_key_rf:
        raise ValueError(
            "API key for Roboflow is not set in environment variables (API_KEY_RF)"
        )

    roboflow_config = RoboflowConfig(
        api_key=api_key_rf,
        workspace=args.workspace_rf,
        project_name=args.project_name_rf,
        version=args.version_rf,
        data_type=args.data_type_rf,
    )

    model_config = ModelConfig(
        train_perc=args.train_perc,
        test_perc=args.test_perc,
        val_perc=args.val_perc,
        keep_perc=args.keep_perc,
        piece_perc=args.piece_perc,
        iters=args.iters,
        fib_flag=args.fib_flag,
        prev_num=args.prev_num,
        threshold=args.threshold,
        class_num=args.class_num,
    )

    return roboflow_config, model_config


if __name__ == "__main__":
    roboflow_config, model_config = parse_args()

    setup_environment()
    locale.getpreferredencoding = lambda: "UTF-8"

    path_to_dataset = download_dataset(
        roboflow_config.api_key,
        roboflow_config.workspace,
        roboflow_config.project_name,
        roboflow_config.version,
        roboflow_config.data_type,
    )

    # Assuming the YAML file is named 'config.yaml' and located in the dataset directory
    path_to_yaml = os.path.join(path_to_dataset, "data.yaml")

    PATH_TO_MODEL = wget.download(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt"
    )

    exp_inc = SizeToMetricModel(
        PATH_TO_MODEL,
        path_to_yaml,
        model_config.train_perc,
        model_config.test_perc,
        model_config.val_perc,
    )

    result_dict, cls_tl_dict = exp_inc.increm_learning_one_class(
        model_config.class_num,
        model_config.keep_perc,
        model_config.iters,
        model_config.piece_perc,
        model_config.fib_flag,
        model_config.prev_num,
        model_config.threshold,
    )
    # Save result_dict and cls_tl_dict to a file
    save_filename = "results.json"
    save_processed_data(model_config.class_num, result_dict, cls_tl_dict)
