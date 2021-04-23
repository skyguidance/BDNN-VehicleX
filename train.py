import torch
import torch.nn.functional as F
import os
from utils.file_utils import save_checkpoint


def train_model(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler):
    epochs = int(config["train"]["epochs"])
    best_top1 = -1
    best_top5 = -1
    best_epoch_top1 = -1
    best_epoch_top5 = -1
    for epoch in range(epochs):
        config["logger"].info(f"Starting Epoch {epoch}.")
        # Train iteration.
        model.train()
        for (i, (x, y)) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            config["logger"].info(
                "Epoch: {}/{} Iteration: {}/{} Loss: {}".format(epoch, epochs, i, len(train_dataloader), loss.item()))
        # Validation
        model.eval()
        config["logger"].info("Evaluating on Validation Set...")
        correct_top1 = 0
        correct_top5 = 0
        total = len(val_dataloader.dataset)
        for (x, y) in val_dataloader:
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
        config["logger"].info("Top-1 Acc. {} Top-5 Acc. {}".format(correct_top1, correct_top5))
        # Save Best Model
        if correct_top1 > best_top1:
            save_path = os.path.join(config["train"]["save_dir"], config["train"]["task"])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pth_path = os.path.join(save_path, "best_top1.pth")
            save_checkpoint(pth_path, model, epoch, optimizer, params=config)
            config["logger"].info("Saved Top-1 Model.{}".format(pth_path))
            best_top1 = correct_top1
            best_epoch_top1 = epoch
        if correct_top5 > best_top5:
            save_path = os.path.join(config["train"]["save_dir"], config["train"]["task"])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pth_path = os.path.join(save_path, "best_top5.pth")
            save_checkpoint(pth_path, model, epoch, optimizer, params=config)
            config["logger"].info("Saved Top-5 Model.{}".format(pth_path))
            best_top5 = correct_top5
            best_epoch_top5 = epoch
        config["logger"].info("Best Top-1 Epoch: {} Best Top-5 Epoch: {}".format(best_epoch_top1, best_epoch_top5))
