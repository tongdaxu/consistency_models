import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from torchvision import transforms
from fid import fid_pytorch, cal_psnr
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from dataset import FFHQDataset
from torch.utils.data import DataLoader
import lpips


def print_avgs(avgs, a=0):
    for key in avgs.keys():
        print("{0}: {1:.4}, ".format(key, np.mean(avgs[key][a:])), end="")
    print("")

label_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dps/roomlayout/label'
predi_folder = '/NEW_EDS/JJ_Group/xutd/consistency_models/results_lsun_bedroom/dps/roomlayout/recon'

transform = transforms.Compose([transforms.ToTensor()])

label_set = FFHQDataset(label_folder, transform)
predi_set = FFHQDataset(predi_folder, transform)

label_loader = list(DataLoader(label_set, 1))
predi_loader = list(DataLoader(predi_set, 1))

fid_computer = fid_pytorch()
fid_computer_256 = fid_pytorch()
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

kid_computer = KernelInceptionDistance(subsets =100, subset_size=500, normalize=True).cuda()

avgs = {
    "mse": [], "psnr": [],
    "fid": [],
    "lpips": [], "fid_256": [],
    "kid": [], "kid_std": [],
    "msssim": []
}

with torch.no_grad():
    fid_computer.clear_pools()
    fid_computer_256.clear_pools()
    for i, (x, x_hat) in tqdm(enumerate(zip(label_loader, predi_loader))):

        x1 = x.cuda()
        x2 = x_hat.cuda()

        unfold = nn.Unfold(kernel_size=(64, 64),stride=(64, 64))
        x1_unfold = unfold(x1).reshape(1, 3, 64, 64, -1)
        x1_unfold = torch.permute(x1_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)
        x2_unfold = unfold(x2).reshape(1, 3, 64, 64, -1)
        x2_unfold = torch.permute(x2_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)
        fid_computer.add_ref_img(x1_unfold)
        fid_computer.add_dis_img(x2_unfold)

        fid_computer_256.add_ref_img(x1)
        fid_computer_256.add_dis_img(x2)

        kid_computer.update(x1_unfold, real=True)
        kid_computer.update(x2_unfold, real=False)

        avgs['mse'].append(torch.mean((x1 - x2)**2).item())
        avgs['psnr'].append(cal_psnr(x1, x2))
        avgs['lpips'].append(loss_fn_alex(x1*2.0-1.0, x2*2.0-1.0).item())

    avgs['fid'].append(fid_computer.summary_pools())
    avgs['fid_256'].append(fid_computer_256.summary_pools())
    kid_mean, kid_std = kid_computer.compute()
    avgs['kid'].append(kid_mean.cpu())
    avgs['kid_std'].append(kid_std.cpu())
    print_avgs(avgs)



