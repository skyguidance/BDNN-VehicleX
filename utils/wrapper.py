import torch.optim
import yaml
import logging
from network.baseline import SimpleNN
from network.BDNN import BiDirectionalNN


def config_wrapper(config_file):
    """
    Config Wrapper
    Load YAML config. Add other utils or tools to the config var.
    :param config_file: The file location of the YAML config file.
    :return: Wrapped config dict.
    """
    config = yaml.safe_load(open(config_file))
    # Add Logger
    config["logger"] = generate_logger()
    config["logger"].info("Successfully loaded YAML config.")
    return config


def generate_logger():
    logger = logging.getLogger("main Logger")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("main_logger.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def network_wrapper(config):
    if config["train"]["task"] == "baseline":
        return SimpleNN(mlp_layers=config["model"]["baseline"]["mlp_layers"],
                        input_dims=config["model"]["input_dims"],
                        output_dims=config["model"]["output_dims"],
                        dropout_rate=config["model"]["baseline"]["dropout"],
                        do_BN=bool(config["model"]["baseline"]["do_BN"]))
    elif config["train"]["task"] == "BDNN":
        return None
    else:
        raise NotImplementedError


def optimizer_wrapper(model, config):
    if config["train"]["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config["train"]["learning_rate"],
                               momentum=config["train"]["momentum"])
    elif config["train"]["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    raise NotImplementedError
