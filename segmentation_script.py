import yaml
import torchvision.transforms as transforms
import torchvision
from ops import get_operator, get_dataset, get_dataloader
from roomsegmentation import visualize_result
import os

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

cfg = load_yaml("/NEW_EDS/JJ_Group/zhuzr/xutd_cm/segmentation_config.yaml")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = get_dataset(**cfg['data'], transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
operator = get_operator(device='cuda', **cfg['operator'])

for i, ref_img in enumerate(loader):
    ref_img = ref_img.to('cuda')
    low_out1 = (operator.forward(ref_img, mode='init') + 1.0) / cfg['nclass']
    torchvision.utils.save_image(low_out1,os.path.join('/NEW_EDS/JJ_Group/zhuzr/icml24/results/roomsegmentation/1_channel_label', str(i).zfill(5)+'.png'))
    low_out2 = (operator.forward(ref_img, mode='noninit') + 1.0) / cfg['nclass']
    visualize_result(low_out2,os.path.join('/NEW_EDS/JJ_Group/zhuzr/icml24/results/roomsegmentation/3_channels_label', str(i).zfill(5)+'.png'))
    
