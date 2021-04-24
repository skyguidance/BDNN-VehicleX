import torch
from torch import nn
import argparse

from utils.wrapper import config_wrapper, network_wrapper, optimizer_wrapper, scheduler_wrapper
from utils.file_utils import dump_buffer_as_pkl
from utils.data_utils import generate_dataloader
from train import train_model
from test import model_benchmark
from utils.plot_graph import plot_graph

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
    model = network_wrapper(config, device)
    optimizer = optimizer_wrapper(model, config)
    criterion = nn.CrossEntropyLoss()
    scheduler = scheduler_wrapper(optimizer, config)
    # Train
    train_model(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler)
    dump_buffer_as_pkl(config)
    plot_graph(config)
    # Test
    model_benchmark(config, test_dataloader, device, model)
