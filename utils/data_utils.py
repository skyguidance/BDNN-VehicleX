import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.cElementTree as ET


class VehicleXDataset(Dataset):
    """VehicleX Dataset."""

    def __init__(self, root_dir, phase):
        """
        :param root_dir: The vehicle-x folder location to index.
        :param phase: either train,val, or test
        """
        self.root_dir = root_dir
        self.phase = phase

        # Check Data Exists.
        if not os.path.exists(os.path.join(root_dir, "finegrained_label.xml")):
            raise FileNotFoundError("Not a valid Vehicle-X Dataset for {}".format(root_dir))

        # Load Resnet Features
        npy_folder = os.path.join(root_dir, phase)
        npy_sample_name = os.listdir(npy_folder)
        self.sample = dict()
        for i in npy_sample_name:
            self.sample[i] = {"loc": os.path.join(npy_folder, i)}

        # Load XML Features
        tree = ET.ElementTree(file=os.path.join(root_dir, "finegrained_label.xml"))
        xml_samples = list(list(tree.getroot())[0])
        for i in xml_samples:
            i = i.attrib
            if i["imageName"] not in self.sample.keys():
                continue
            self.sample[i["imageName"]].update(i)

        self.sample = list(self.sample.values())

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        sample = self.sample[item]

        features = np.load(sample["loc"])
        # Total 12 Colors.
        color = np.zeros(12)
        color[int(sample["colorID"])] = 1
        features = np.append(features, color)
        # Total 11 Types of vehicle.
        types = np.zeros(11)
        types[int(sample["typeID"])] = 1
        features = np.append(features, types)
        # Total 1362 Classes + 1 Extra Node (For 1to1 BDNN mapping)
        gt = int(sample["vehicleID"]) - 1
        uid = int(sample["cameraID"][1:] + sample["imageName"][:-4].split("_")[-1]) / 100000
        return torch.Tensor(features), torch.tensor(gt).long(), torch.tensor(uid).float()


def generate_dataloader(config):
    root_dir = config["data"]["root_dir"]
    train_dataset = VehicleXDataset(root_dir, phase="train")
    val_dataset = VehicleXDataset(root_dir, phase="val")
    test_dataset = VehicleXDataset(root_dir, phase="test")

    config["logger"].info("Generating Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, drop_last=True,
                                  pin_memory=True, num_workers=config["train"]["num_workers"])
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"],
                                num_workers=config["train"]["num_workers"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"],
                                 num_workers=config["train"]["num_workers"])

    return (train_dataloader, val_dataloader, test_dataloader)
