import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def plot_graph(config, config2):
    # Plot loss graph.
    epoch1 = np.arange(1, len(config["buffer"]["loss"]) + 1, 1)
    epoch2 = np.arange(1, len(config2["buffer"]["loss"]) + 1, 1)
    loss1 = config["buffer"]["loss"]
    loss2 = config2["buffer"]["loss"]
    plt.figure()
    plt.plot(epoch1, loss1, color='r', label="Simple NN Loss")
    plt.plot(epoch2, loss2, color='g', label="Simple NN BN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join("loss.jpg"))
    top1_acc1 = config["buffer"]["top1_acc"]
    top5_acc1 = config["buffer"]["top5_acc"]
    top1_acc2 = config2["buffer"]["top1_acc"]
    top5_acc2 = config2["buffer"]["top5_acc"]
    plt.figure()
    plt.plot(epoch1, top1_acc1, color='r', label="Simple NN Top-1 Accuracy")
    plt.plot(epoch1, top5_acc1, color='g', label="Simple NN Top-5 Accuracy")
    plt.plot(epoch2, top1_acc2, color='b', label="Simple NN BN Top-1 Accuracy")
    plt.plot(epoch2, top5_acc2, color='y', label="Simple NN BN Top-5 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(os.path.join("accuracy.jpg"))


if __name__ == '__main__':
    config = pickle.load(
        open("/Users/tianyiqi/Documents/Sem 1 2021/COMP8420/Assignment1/document/buffer/BDNN-5Layer/config_buffer.pkl",
             "rb"))
    config2 = pickle.load(
        open("/Users/tianyiqi/Documents/Sem 1 2021/COMP8420/Assignment1/document/buffer/BDNN-5Layer-BN/config_buffer.pkl",
             "rb"))
    plot_graph(config,config2)