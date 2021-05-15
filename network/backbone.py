import torch
import torch.nn as nn
import timm
import numpy as np


def get_self_implemented_backbone(config):
    raise NotImplementedError


def get_backbone(config):
    """
    Generate backbone
    :param config: config file
    :return: backbone network, and feature output dim for backbone.
    """
    if config["model"]["backbone"]["net"] in timm.list_models():
        backbone = timm.create_model(config["model"]["backbone"]["net"],
                                     pretrained=config["model"]["backbone"]["pretrained"],
                                     num_classes=0)
        input_dim = backbone.default_cfg["input_size"]
        backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove the classification layer.
    else:
        backbone, input_dim = get_self_implemented_backbone(config)
    # Getting output dim
    backbone.eval()
    backbone.cpu()
    dummy_input = torch.from_numpy(np.zeros(input_dim)).unsqueeze(0).float()
    dumpy_output = backbone(dummy_input)
    return backbone, dumpy_output.shape[1]
