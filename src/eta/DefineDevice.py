import os

import torch
from dotenv import load_dotenv


def define_device():
    """
    Definition of a device for performing calculations.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    return device_name


def setup_env(current_path):
    """
    Set up the environment for the path of a calling file.
    Parameters:
        current_path (str | Path): The current path where the environment is being set up.
    Returns:
        None
    """

    DEVICE_NAME = define_device()

    if DEVICE_NAME == "cuda":
        env_path = current_path / "cuda.env"

    elif DEVICE_NAME == "mps":
        env_path = current_path / "mps.env"
    else:
        env_path = current_path / "cpu.env"

    load_dotenv(env_path)

    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    return
