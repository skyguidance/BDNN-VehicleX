import numpy as np
import pandas as pd
import torch


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = preds.eq(labels)
    acc = correct.float().mean().item()
    return acc
