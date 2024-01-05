from dataset import FFHQDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm
import numpy as np

label_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dps/roomlayout/label'
predi_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dps/roomlayout/recon'

transform = transforms.Compose([transforms.ToTensor()])

label_set = FFHQDataset(label_folder, transform)
predi_set = FFHQDataset(predi_folder, transform)

label_loader = list(DataLoader(label_set, 1))
predi_loader = list(DataLoader(predi_set, 1))

fids = []
kids = []
lpips = []
mses = []
for i in tqdm(range(len(label_loader))):
    pass
