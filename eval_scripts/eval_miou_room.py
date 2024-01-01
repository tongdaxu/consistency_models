from dataset import FFHQDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm
import numpy as np

label_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dpscm2/roomlayout/input'
predi_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dpscm2/roomlayout/low_res'

transform = transforms.Compose([transforms.ToTensor()])

label_set = FFHQDataset(label_folder, transform)
predi_set = FFHQDataset(predi_folder, transform)

label_loader = list(DataLoader(label_set, 1))
predi_loader = list(DataLoader(predi_set, 1))

colors = torch.tensor([
    [0.9764706, 0.27058825, 0.3647059], [1., 0.8980392, 0.6666667],
    [0.5647059, 0.80784315, 0.70980394], [0.31764707, 0.31764707, 0.46666667],
    [0.94509804, 0.96862745, 0.8235294]])
colors = colors * 255
colors = colors.to(torch.int32)

jaccard = JaccardIndex(task='multiclass', num_classes=5)

def rgb2label(x, palette):
    b, _, h, w = x.shape
    out = torch.zeros([b,h,w])
    for i in range(palette.shape[0]):
        ind = (x == palette[i][None, :, None, None]).to(torch.int32)[:,0]
        out += ind * (i)
    return out
mious = []
for i in tqdm(range(len(label_loader))):
    label_img = (label_loader[i] * 255).to(torch.int32)
    predi_img = (predi_loader[i] * 255).to(torch.int32)
    label_out = rgb2label(label_img, colors)
    predi_out = rgb2label(predi_img, colors)
    miou = jaccard(label_out, predi_out)
    mious.append(miou)

print("miou: {0:.4f}".format(np.mean(mious)))
