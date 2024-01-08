import torch
from torchmetrics.multimodal.clip_score import CLIPScore
import yaml
import torchvision.transforms as transforms
from dataset import FFHQDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

label_folder = '/NEW_EDS/JJ_Group/zhuzr/icml24/results/roomtext/label'
predi_folder = '/NEW_EDS/JJ_Group/zhuzr/icml24/results/roomtext/recon'

transform = transforms.Compose([transforms.ToTensor()])

label_set = FFHQDataset(label_folder, transform)
predi_set = FFHQDataset(predi_folder, transform)

label_loader = list(DataLoader(label_set, 1))
predi_loader = list(DataLoader(predi_set, 1))

metric = CLIPScore(model_name_or_path="/NEW_EDS/JJ_Group/zhuzr/huggingface/openai/clip-vit-large-patch14")
score_collection = []

file = open("/NEW_EDS/JJ_Group/zhuzr/icml24/results/label_caption")
file_content = file.readlines()

for i,img in enumerate(label_loader):
    img = (img * 255).int()
    score = metric(img * 255, file_content[i].split('\'')[1])
    score = score.detach().numpy()
    score_collection.append(score)
    print(i)
    print(np.mean(score_collection))
print(np.mean(score_collection))
    
