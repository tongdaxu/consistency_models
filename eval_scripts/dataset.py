from abc import ABC, abstractmethod
from functools import partial
import yaml

import torch.nn as nn
from torch.nn import functional as F

import torchvision

import numpy as np
import torch
from math import pi

from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os 
from torchvision import transforms

class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return min(len(self.fpaths), 1000)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
