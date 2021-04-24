import torch
from torch import nn
import torch.nn.functional as F
import os
from utils.file_utils import save_checkpoint
from collections import OrderedDict


def train_model_baseline(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler):
    epochs = int(config["train"]["epochs"])
    best_top1 = -1
    best_top5 = -1
    best_epoch_top1 = -1
    best_epoch_top5 = -1
    for epoch in range(epochs):
        config["logger"].info(f"Starting Epoch {epoch}.")
        # Train iteration.
        model.train()
        for (i, (x, y, extra)) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            config["logger"].info(
                "Epoch: {}/{} Iteration: {}/{} Loss: {}".format(epoch, epochs, i, len(train_dataloader), loss.item()))
        if scheduler:
            scheduler.step()
        # Validation
        model.eval()
        config["logger"].info("Evaluating on Validation Set...")
        correct_top1 = 0
        correct_top5 = 0
        total = len(val_dataloader.dataset)
        for (x, y, extra) in val_dataloader:
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


def train_model_BDNN(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler):
    epochs = int(config["train"]["epochs"])
    best_top1 = -1
    best_top5 = -1
    best_epoch_top1 = -1
    best_epoch_top5 = -1
    for epoch in range(epochs):
        config["logger"].info(f"Starting Epoch {epoch}.")
        # Train iteration.
        model_F, model_R = model
        model_F.train()
        model_R.train()
        config["logger"].info("BDNN Forward ...")
        for (i, (x, y, extra)) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            extra = extra.to(device)
            y_pred, extra_pred = model_F(x)
            loss_classfication = criterion(y_pred, y)
            loss_extra = F.mse_loss(extra_pred, extra)
            loss = loss_classfication + loss_extra
            optimizer[0].zero_grad()
            loss.backward()
            optimizer[0].step()
            config["logger"].info(
                "[F] Epoch: {}/{} Iteration: {}/{} Loss: {}".format(epoch, epochs, i, len(train_dataloader),
                                                                    loss.item()))

        config["logger"].info("Syncing Weights ...")
        state_dict = model_F.state_dict()
        transformed_state_dict = OrderedDict()
        max_mlp_layer_index = len(config["model"]["BDNN"]["mlp_layers"]) - 1
        for k, v in state_dict.items():
            # Copy Weights
            if ("weight" in k) and (len(v.shape) == 2):
                transformed_state_dict[k] = v.T
            # Copy Bias
            elif ("bias" in k) and (f"layer{max_mlp_layer_index}" in k):
                transformed_state_dict[f"net.final.bias"] = v
            elif ("bias" in k) and ("final" not in k):
                current_layer_index = int(k.split(".")[1].replace("layer", ""))
                transformed_state_dict[f"net.layer{current_layer_index + 1}.bias"] = v
        model_R.load_state_dict(transformed_state_dict, strict=False)

        config["logger"].info("BDNN Backward ...")
        for (i, (x, y, extra)) in enumerate(train_dataloader):
            x = x.to(device)
            # y need one-hot encoding for reverse.
            one_hot_pool = torch.eye(config["model"]["output_dims"] - 1)
            y_one_hot = one_hot_pool[y]
            y_combined = torch.cat((y_one_hot, extra.reshape(-1, 1)), dim=1)
            y_combined = y_combined.to(device)
            x_pred = model_R(y_combined)
            loss = F.mse_loss(x_pred, x)
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[1].step()
            config["logger"].info(
                "[B] Epoch: {}/{} Iteration: {}/{} Loss: {}".format(epoch, epochs, i, len(train_dataloader),
                                                                    loss.item()))
        config["logger"].info("Syncing Weights ...")
        state_dict = model_R.state_dict()
        transformed_state_dict = OrderedDict()
        max_mlp_layer_index = len(config["model"]["BDNN"]["mlp_layers"]) - 1
        for k, v in state_dict.items():
            # Copy Weights
            if ("weight" in k) and (len(v.shape) == 2):
                transformed_state_dict[k] = v.T
            # Copy Bias
            elif ("bias" in k) and ("final" in k):
                transformed_state_dict[f"net.layer{max_mlp_layer_index}.bias"] = v
            elif ("bias" in k) and ("layer0" not in k):
                current_layer_index = int(k.split(".")[1].replace("layer", ""))
                transformed_state_dict[f"net.layer{current_layer_index - 1}.bias"] = v
        model_F.load_state_dict(transformed_state_dict, strict=False)

        if scheduler:
            scheduler[0].step()
            scheduler[1].step()
        # Validation
        model_F.eval()
        config["logger"].info("Evaluating on Validation Set...")
        correct_top1 = 0
        correct_top5 = 0
        total = len(val_dataloader.dataset)
        for (x, y, extra) in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = F.softmax(model_F(x)[0], dim=1)
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
            save_checkpoint(pth_path, model_F, epoch, optimizer[0], params=config)
            config["logger"].info("Saved Top-1 Model.{}".format(pth_path))
            best_top1 = correct_top1
            best_epoch_top1 = epoch
        if correct_top5 > best_top5:
            save_path = os.path.join(config["train"]["save_dir"], config["train"]["task"])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pth_path = os.path.join(save_path, "best_top5.pth")
            save_checkpoint(pth_path, model_F, epoch, optimizer[0], params=config)
            config["logger"].info("Saved Top-5 Model.{}".format(pth_path))
            best_top5 = correct_top5
            best_epoch_top5 = epoch
        config["logger"].info("Best Top-1 Epoch: {} Best Top-5 Epoch: {}".format(best_epoch_top1, best_epoch_top5))


def train_model(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler):
    if config["train"]["task"] == "baseline":
        train_model_baseline(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler)
    elif config["train"]["task"] == "BDNN":
        train_model_BDNN(config, train_dataloader, val_dataloader, device, model, optimizer, criterion, scheduler)
    else:
        raise NotImplementedError
