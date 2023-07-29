import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def read_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines if len(line.strip()) > 0]
    return txt_data


class BreastDataset(Dataset):
    def __init__(self, root_dir, txt_b, txt_m, transform=None):
        super(BreastDataset, self).__init__()
        self.root_dir = root_dir
        self.txt_path = [txt_b, txt_m]
        self.classes = ['benign', 'malignant']
        self.maskes = ['benign_mask', 'malignant_mask']
        self.num_cls = len(self.classes)
        self.img_list = []
        self.mask_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            mask_list = [os.path.join(self.root_dir, self.maskes[c], item.split(".")[0] + "_mask.png") for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
            self.mask_list += mask_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.img_list[idx][0])
        mask = Image.open(self.mask_list[idx])
        manual = np.array(mask)
        for i in range(manual.shape[0]):
            for j in range(manual.shape[1]):
                if manual[i, j] > 128:
                   manual[i, j] = 255
                else:
                    manual[i, j] = 0
        mask = Image.fromarray(manual / 255)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        sample = {'img': image,
                  'label': int(self.img_list[idx][1]),
                  'mask': mask}

        return sample
