import torch
import numpy as np
from tqdm import tqdm
import os

from utils.data_utils import generate_dataloader
from utils.file_utils import load_checkpoint


def generate_numpy_featues(config, device, model):
    """
    Generate Vehicle-X like numpy feature for additional tasks.
    """
    config["logger"].info("<<<<<<<>>>>>>><<<<<<<>>>>>>><<<<<<<>>>>>>>")
    config["logger"].info("Generate Numpy Features for CNN...")
    config["buffer"]["stage"] = "CNN_numpy_generate"
    # Genreate Dataloader
    train_dataloader, val_dataloader, test_dataloader = generate_dataloader(config)
    config["logger"].info("Loading Top-1 Model...")
    model_path = os.path.join(config["test"]["model_dir"], config["test"]["task"],
                              config["test"]["CNN_additional"]["net"], "best_top1.pth")
    load_checkpoint(model_path, model)
    model.eval()
    config["logger"].info("Generate Train Features...")
    for (i, x, y, extra, path) in tqdm(enumerate(train_dataloader)):
        x = x["image"].to(device)
        model = model.to(device)
        x = model.forward_backbone(x)
        x = x.cpu().detach().numpy()[0]
        save_folder = os.path.join(config["test"]["model_dir"], config["test"]["task"],
                                   config["test"]["CNN_additional"]["net"], "numpy_features", "train")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, path.replace(".jpg", ".npy"))
        np.save(save_path, x)
    config["logger"].info("Generate Val Features...")
    for (i, x, y, extra, path) in tqdm(enumerate(val_dataloader)):
        x = x["image"].to(device)
        model = model.to(device)
        x = model.forward_backbone(x)
        x = x.cpu().detach().numpy()[0]
        save_folder = os.path.join(config["test"]["model_dir"], config["test"]["task"],
                                   config["test"]["CNN_additional"]["net"], "numpy_features", "val")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, path.replace(".jpg", ".npy"))
        np.save(save_path, x)
    config["logger"].info("Generate Test Features...")
    for (i, x, y, extra, path) in tqdm(enumerate(train_dataloader)):
        x = x["image"].to(device)
        model = model.to(device)
        x = model.forward_backbone(x)
        x = x.cpu().detach().numpy()[0]
        save_folder = os.path.join(config["test"]["model_dir"], config["test"]["task"],
                                   config["test"]["CNN_additional"]["net"], "numpy_features", "test")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, path.replace(".jpg", ".npy"))
        np.save(save_path, x)
