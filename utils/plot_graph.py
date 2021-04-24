import matplotlib.pyplot as plt
import numpy as np
import os


def plot_graph(config):
    # Plot loss graph.
    epoch = np.arange(1, len(config["buffer"]["loss"]) + 1, 1)
    loss = config["buffer"]["loss"]
    plt.figure()
    plt.plot(epoch, loss, color='r', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(config["train"]["save_dir"], config["train"]["task"], "loss.jpg"))
    top1_acc = config["buffer"]["top1_acc"]
    top5_acc = config["buffer"]["top5_acc"]
    plt.figure()
    plt.plot(epoch, top1_acc, color='r', label="Top-1 Accuracy")
    plt.plot(epoch, top5_acc, color='g', label="Top-5 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(os.path.join(config["train"]["save_dir"], config["train"]["task"], "accuracy.jpg"))
