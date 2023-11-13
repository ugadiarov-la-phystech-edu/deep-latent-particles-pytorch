import glob
import os

import cv2
import torch
from torch.utils.data import Dataset


class Shapes2dDataset(Dataset):
    def __init__(
        self, path: str, size: int = None
    ):
        super().__init__()
        self.path = path
        files = glob.glob(os.path.join(self.path, '*'))
        if size is None:
            self.size = len(files)
        else:
            self.size = size

        self.files = files[:self.size]

    def __getitem__(self, index: int):
        image_path = self.files[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = torch.as_tensor(image, dtype=torch.float32) / 255.
        return torch.permute(image, (2, 0, 1))

    def __len__(self):
        return len(self.files)
