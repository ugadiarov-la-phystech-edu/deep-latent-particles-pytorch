import glob
import os

import cv2
import torch
from torch.utils.data import Dataset


class Shapes2dDataset(Dataset):
    def __init__(
        self, path: str, image_size: int = None, length: int = None
    ):
        super().__init__()
        self.path = path
        self.image_size = image_size
        files = glob.glob(os.path.join(self.path, '*'))
        if length is None:
            self.length = len(files)
        else:
            self.length = length

        self.files = files[:self.length]

    def __getitem__(self, index: int):
        image_path = self.files[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        image = torch.as_tensor(image, dtype=torch.float32) / 255.
        return torch.permute(image, (2, 0, 1))

    def __len__(self):
        return len(self.files)
