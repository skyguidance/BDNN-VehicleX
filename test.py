import os

import torch
import torch.nn.functional as F
from utils.file_utils import load_checkpoint


def test_model(config, test_dataloader, device, model):
    model = model.to(device)
    model.eval()
    config["logger"].info("Testing Model...")
    correct_top1 = 0
    correct_top5 = 0
    total = len(test_dataloader.dataset)
    for (x, y) in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = F.softmax(model(x), dim=1)
        y_pred = y_pred.cpu()
        y = y.cpu()
        # Calc Top-1
        pred = y_pred.argmax(dim=1)
        correct_top1 += torch.eq(pred, y).sum().float().item()
        # Calc Top-5
        maxk = max((1, 5))
        y_resize = y.view(-1, 1)
        _, pred = y_pred.topk(maxk, 1, True, True)
        correct_top5 += torch.eq(pred, y_resize).sum().float().item()
    correct_top1 /= total
    correct_top5 /= total
    config["logger"].info("============Testing Results============")
    config["logger"].info("Top-1 Acc. {} Top-5 Acc. {}".format(correct_top1, correct_top5))


def model_benchmark(config, test_dataloader, device, model):
    config["logger"].info("Model Testing...")
    model = model.cpu()
    if bool(config["test"]["test_top_1"]):
        config["logger"].info("Loading Top-1 Model...")
        model_path = os.path.join(config["test"]["model_dir"], config["test"]["task"], "best_top1.pth")
        load_checkpoint(model_path, model)
        test_model(config, test_dataloader, device, model)

    if bool(config["test"]["test_top_5"]):
        config["logger"].info("Loading Top-5 Model...")
        model_path = os.path.join(config["test"]["model_dir"], config["test"]["task"], "best_top5.pth")
        load_checkpoint(model_path, model)
        test_model(config, test_dataloader, device, model)
