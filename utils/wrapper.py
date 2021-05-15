import torch.optim
import yaml
import logging
from network.baseline import SimpleNN
from network.BDNN import BiDirectionalNN, BiDirectionalNN_R
from network.CNN import CNN


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
    # Build Buffer to store temporal data.
    config["buffer"] = dict()
    config["buffer"]["loss"] = list()
    config["buffer"]["top1_acc"] = list()
    config["buffer"]["top5_acc"] = list()
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


def network_wrapper(config, device):
    if config["train"]["task"] == "baseline":
        model = SimpleNN(mlp_layers=config["model"]["baseline"]["mlp_layers"],
                         input_dims=config["model"]["input_dims"],
                         output_dims=config["model"]["output_dims"],
                         dropout_rate=config["model"]["baseline"]["dropout"],
                         do_BN=bool(config["model"]["baseline"]["do_BN"]))
        model = model.to(device)
        return model
    elif config["train"]["task"] == "BDNN":
        model_F = BiDirectionalNN(mlp_layers=config["model"]["BDNN"]["mlp_layers"],
                                  input_dims=config["model"]["input_dims"],
                                  output_dims=config["model"]["output_dims"],
                                  dropout_rate=config["model"]["BDNN"]["dropout"],
                                  do_BN=bool(config["model"]["BDNN"]["do_BN"]))
        model_F = model_F.to(device)
        model_R = BiDirectionalNN_R(mlp_layers=config["model"]["BDNN"]["mlp_layers"],
                                    input_dims=config["model"]["input_dims"],
                                    output_dims=config["model"]["output_dims"],
                                    dropout_rate=config["model"]["BDNN"]["dropout"],
                                    do_BN=bool(config["model"]["BDNN"]["do_BN"]))
        model_R = model_R.to(device)
        return model_F, model_R
    elif config["train"]["task"] == "CNN":
        model = CNN(config)
        model = model.to(device)
        return model
    else:
        raise NotImplementedError


def optimizer_wrapper(model, config):
    if config["train"]["task"] == "baseline" or config["train"]["task"] == "CNN":
        if config["train"]["optimizer"] == "SGD":
            return torch.optim.SGD(model.parameters(), lr=config["train"]["learning_rate"],
                                   momentum=config["train"]["momentum"])
        elif config["train"]["optimizer"] == "Adam":
            return torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
        else:
            raise NotImplementedError
    elif config["train"]["task"] == "BDNN":
        if config["train"]["optimizer"] == "SGD":
            optim_F = torch.optim.SGD(model[0].parameters(), lr=config["train"]["learning_rate"],
                                      momentum=config["train"]["momentum"])
            optim_R = torch.optim.SGD(model[1].parameters(), lr=config["train"]["learning_rate"],
                                      momentum=config["train"]["momentum"])
            return optim_F, optim_R
        elif config["train"]["optimizer"] == "Adam":
            optim_F = torch.optim.Adam(model[0].parameters(), lr=config["train"]["learning_rate"])
            optim_R = torch.optim.Adam(model[1].parameters(), lr=config["train"]["learning_rate"])
            return optim_F, optim_R
        else:
            raise NotImplementedError


def scheduler_wrapper(optimizer, config):
    if config["train"]["task"] == "baseline":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80, 130], gamma=0.5)
    elif config["train"]["task"] == "BDNN":
        scheduler_F = torch.optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=[30, 80, 130], gamma=0.5)
        scheduler_R = torch.optim.lr_scheduler.MultiStepLR(optimizer[1], milestones=[30, 80, 130], gamma=0.5)
        return scheduler_F, scheduler_R
    elif config["train"]["task"] == "CNN":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)
    else:
        raise NotImplementedError
