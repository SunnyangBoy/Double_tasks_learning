import torch
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
import os 
import numpy as np
from PIL import Image
import random


class DetectAngleDataset(Dataset):
    def __init__(self, img_rootdir, images, transforms=None):
        self.img_rootdir = img_rootdir
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        assert idx <= self.__len__(), "index out of range"
        img_name = self.images[idx]['name']
        img_dir = os.path.join(self.img_rootdir, img_name)
        img = Image.open(img_dir)
        img = img.convert("L")
        img = img.resize((224, 224))

        if self.transforms != None:
            img = self.transforms(img)

        # img.save(os.path.join('/root/tmp_test/', img_name))

        img = np.array(img)
        img = img / 255
        img = torch.from_numpy(img).float()
        # normalize = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # img = normalize(img)
        img = torch.unsqueeze(img, 0)
        img_class = self.images[idx]['class']
        img_mode = self.images[idx]['mode']
        return (img, img_class, img_mode)


class RandomEnhance:
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, img):
        sub_list = random.sample(self.transforms, 3)
        for t in sub_list:
            img = t(img)
        return img


def img_enhancer(x):
    x = transforms.Resize(size=(224, 224))(x)
    tfs_list = [
        transforms.RandomCrop(size=(224, 224), padding=10, padding_mode='edge'),
        transforms.RandomRotation(10, expand=False),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
    ]
    x = RandomEnhance(tfs_list)(x)
    x = transforms.Resize(size=(224, 224))(x)
    return x
