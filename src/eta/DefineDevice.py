import torch


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
