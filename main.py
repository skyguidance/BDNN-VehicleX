import torch
from torch import nn
import argparse

from utils.wrapper import config_wrapper, network_wrapper, optimizer_wrapper
from dataset.data_utils import generate_dataloader
from train import train_model
from test import test_model


def parse_args():
    parser = argparse.ArgumentParser(description="COMP8420-NNDL Assignment 1")
    parser.add_argument('--config', '-c', required=True, type=str,
                        help='The YAML config filepath.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = config_wrapper(args.config)
    # Build Dataloader
    train_dataloader, val_dataloader, test_dataloader = generate_dataloader(config)
    # Get Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build Model
    model = network_wrapper(config)
    model = model.to(device)
    optimizer = optimizer_wrapper(model, config)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    # Train
    train_model(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler)
    # Test
    test_model(config, test_dataloader, device, model)
