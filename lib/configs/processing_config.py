from dataclasses import dataclass


@dataclass
class RoboflowConfig:
    api_key: str
    workspace: str
    project_name: str
    version: str
    data_type: str


@dataclass
class ModelConfig:
    path_to_yaml: str
    train_perc: float
    test_perc: float
    val_perc: float
    keep_perc: float
    piece_perc: float
    iters: int
    fib_flag: bool
    prev_num: int
    threshold: float
    class_num: int
