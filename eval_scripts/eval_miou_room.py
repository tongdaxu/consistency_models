from dataset import FFHQDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm
import numpy as np

label_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dps/roomlayout/input'
predi_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dps/roomlayout/low_res'

transform = transforms.Compose([transforms.ToTensor()])

label_set = FFHQDataset(label_folder, transform)
predi_set = FFHQDataset(predi_folder, transform)

label_loader = list(DataLoader(label_set, 1))
predi_loader = list(DataLoader(predi_set, 1))

jaccard = JaccardIndex(task='multiclass', num_classes=5)

mious = []
for i in tqdm(range(len(label_loader))):
    label_img = (label_loader[i] * 5).to(torch.int32) - 1
    predi_img = (predi_loader[i] * 5).to(torch.int32) - 1
    miou = jaccard(label_img, predi_img)
    mious.append(miou)

print("miou: {0:.4f}".format(np.mean(mious)))
