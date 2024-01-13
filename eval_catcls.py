# from resnet import resnet50
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from PIL import Image
from glob import glob
from torchvision.datasets import VisionDataset

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm
import numpy as np
from roomsegmentation import SegmentationModule, ModelBuilder
from ops import LinearOperator, FFHQDataset
from typing import Callable, Optional

device = torch.device('cuda:0')

from torchvision.models import resnet50, resnet101,ResNet50_Weights, ResNet101_Weights
from torchvision.models import ViT_B_16_Weights, vit_b_16

# model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).cuda()
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).cuda()
model.eval()
image_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dpscm/catcls2/recon'
# image_folder = '/NEW_EDS/JJ_Group/zhuzr/consistency_models/raw_image4'

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(224), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
image_set = FFHQDataset(image_folder, transform)
image_loader = list(DataLoader(image_set, 1))

acc = []
for i, image in tqdm(enumerate(image_loader)):
    image = image.cuda()
    score = model(image)
    pred = torch.argmax(score)
    if (pred.item() - 281) == i % 5:
        acc.append(1)
    else:
        acc.append(0)
print("acc: {0:.4f}".format(np.mean(acc)))
