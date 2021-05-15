import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.cElementTree as ET
import cv2
from albumentations import Compose, Normalize, IAAAffine, RandomBrightnessContrast, MotionBlur, CLAHE
import timm


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


class VehicleXVer2Dataset(Dataset):
    """VehicleX Ver 2 Dataset"""

    def __init__(self, root_dir, phase, target_size, transform=None, including_path=False):
        """
        :param root_dir: The vehicle-x folder location to index.
        :param phase: either train,val, or test
        :param target_size: Resize image to target size during __getitem__
        :param transform (callable, optional): Optional Transform to be applied on a sample
        :param including_path: return will include the exact image path.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.target_size = target_size
        self.including_path = including_path

        # Check Data Exists.
        if not os.path.exists(os.path.join(root_dir, "Classification Task")):
            raise FileNotFoundError("Not a valid Vehicle-X Ver 2 Dataset for {}".format(root_dir))

        # Prepare image location
        img_folder = os.path.join(root_dir, "Classification Task", phase)
        img_sample_name = os.listdir(img_folder)
        self.sample = dict()
        for i in img_sample_name:
            self.sample[i] = {"loc": os.path.join(img_folder, i)}

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
        # Get IMG
        image = cv2.imread(sample["loc"])
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Do data augmentation
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
        # Get Additional Features
        features = np.array([])
        # Total 12 Colors.
        color = np.zeros(12)
        color[int(sample["colorID"])] = 1
        features = np.append(features, color)
        # Total 11 Types of vehicle.
        types = np.zeros(11)
        types[int(sample["typeID"])] = 1
        features = np.append(features, types)
        # Get labels
        # Total 1362 Classes + 1 Extra Node (For 1to1 BDNN mapping)
        gt = int(sample["vehicleID"]) - 1
        uid = int(sample["cameraID"][1:] + sample["imageName"][:-4].split("_")[-1]) / 100000

        feature = {"image": image, "features": torch.Tensor(features)}
        if not self.including_path:
            return feature, torch.tensor(gt).long(), torch.tensor(uid).float()
        else:
            return feature, torch.tensor(gt).long(), torch.tensor(uid).float(), sample["imageName"]


def generate_transform(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       divide_by=255.0,
                       shear_limit=0,
                       rotate_limit=0,
                       brightness_limit=0.0,
                       contrast_limit=0.0,
                       clahe_p=0.0,
                       blur_p=0.0
                       ):
    """
    Pytorch Transform Generator. Generate both train data augmentation and valid/test data augmentation.
    :param mean: The mean value to normalize (standard ImageNet value)
    :param std: The standard value to normalize (standard ImageNet value)
    :param divide_by: The max value of pixel.
    :param shear_limit: data augmentation: shear rage.
    :param rotate_limit: data augmentation: rotate rage.
    :param brightness_limit: data augmentation: brightness rage.
    :param contrast_limit: data augmentation: contrast rage.
    :param clahe_p: data augmentation: clahe probability.
    :param blur_p: data augmentation: blur probability (motion blur).
    :return: Composed of train_transform and evaluation transform
    """
    if mean is not None and std is not None:
        norm = Normalize(mean=mean, std=std, max_pixel_value=divide_by)
    else:
        # Standard ImageNet Parameter.
        norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=divide_by)
    train_transform = Compose([
        IAAAffine(rotate=(-rotate_limit, rotate_limit), shear=(-shear_limit, shear_limit), mode='constant'),
        RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit),
        MotionBlur(p=blur_p),
        CLAHE(p=clahe_p),
        norm,
    ])
    eval_transform = Compose([norm])

    return train_transform, eval_transform


def generate_dataloader(config):
    config["logger"].info("Detect Dataset Version...")
    is_v2 = False
    if "is_v2" in config["data"].keys() and config["data"]["is_v2"]:
        config["logger"].info("Dataset Configurator is set to Version 2")
        is_v2 = True
    else:
        config["logger"].info("Dataset Configurator is set to Version 1")
    root_dir = config["data"]["root_dir"]

    if is_v2:
        # If is pretrained_network, get pre-trained network prarameter.
        transform_mean, transform_std, input_size = (None, None, None)
        if config["model"]["backbone"]["pretrained"]:
            config["logger"].info("Getting Pretained Network Config...")
            # Only get the default training config. No need to generate a weights filled real network.
            model = timm.create_model(config["model"]["backbone"]["net"], pretrained=False)
            input_size = model.default_cfg["input_size"][1:]  # First dim is channel.
            transform_mean = model.default_cfg["mean"]
            transform_std = model.default_cfg["std"]
            del model
        train_transform, eval_transform = generate_transform(mean=transform_mean, std=transform_std)
        if "stage" in config["buffer"].keys() and config["buffer"]["stage"] == "CNN_numpy_generate":
            # CNN Feature Generator Called Me.
            train_dataset = VehicleXVer2Dataset(root_dir, target_size=input_size, phase="train",
                                                transform=eval_transform, including_path=True)
            val_dataset = VehicleXVer2Dataset(root_dir, target_size=input_size, phase="val", transform=eval_transform,
                                              including_path=True)
            test_dataset = VehicleXVer2Dataset(root_dir, target_size=input_size, phase="test", transform=eval_transform,
                                               including_path=True)
        else:
            # Standard train/val process
            train_dataset = VehicleXVer2Dataset(root_dir, target_size=input_size, phase="train",
                                                transform=train_transform)
            val_dataset = VehicleXVer2Dataset(root_dir, target_size=input_size, phase="val", transform=eval_transform)
            test_dataset = VehicleXVer2Dataset(root_dir, target_size=input_size, phase="test", transform=eval_transform)
    else:
        train_dataset = VehicleXDataset(root_dir, phase="train")
        val_dataset = VehicleXDataset(root_dir, phase="val")
        test_dataset = VehicleXDataset(root_dir, phase="test")
    config["logger"].info("Generating Dataloader...")
    if "stage" in config["buffer"].keys() and config["buffer"]["stage"] == "CNN_numpy_generate":
        # CNN Feature Generator Called Me.
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=config["train"]["num_workers"])
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=config["train"]["num_workers"])
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=config["train"]["num_workers"])
    else:
        # Standard train/val process
        train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True,
                                      drop_last=True,
                                      pin_memory=True, num_workers=config["train"]["num_workers"])
        val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"],
                                    num_workers=config["train"]["num_workers"])
        test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"],
                                     num_workers=config["train"]["num_workers"])

    return (train_dataloader, val_dataloader, test_dataloader)
