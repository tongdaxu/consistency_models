"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import yaml
import torchvision.transforms as transforms
import torchvision
import torch

from cm import logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample, karras_inverse
from ops import get_operator, get_dataset, get_dataloader
device = th.device('cuda:0')

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=False,
    )

    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.distiller_path == "":
        distiller = None
    else:
        distiller, _ = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()),
            distillation=True,
        )
        distiller.load_state_dict(
            th.load(args.distiller_path, map_location="cpu")
        )
        distiller.to(device)
        if args.use_fp16:
            distiller.convert_to_fp16()
        distiller.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    cfg = load_yaml(args.cfg)
    zeta = cfg['zeta']
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    dataset = get_dataset(**cfg['data'], transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    operator = get_operator(device=device, **cfg['operator'])
    save_dir = args.savedir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, cfg['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'low_res', 'E0t', 'x0t', 'reE0t', 'rex0t']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    for i, ref_img in enumerate(loader):
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        y_n = operator.forward(ref_img)
        model_kwargs = {}
        sample = karras_inverse(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            y = y_n,
            operator = operator,
            zeta = zeta,
            model_kwargs=model_kwargs,
            device=device,
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            distiller=distiller,
            save_dir=out_path,
        )
        if cfg['operator']['name'] == 'roomlayout':
            from roomlayout import label_as_rgb_visual
            torchvision.utils.save_image(label_as_rgb_visual(th.max(y_n, 1)[1]), os.path.join(out_path, 'input', fname))
            # avoid overfitting in evaluation
            # trans = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            # sample_flip = trans((sample + 1.0) / 2.0) * 2.0 - 1.0
            sample_flip = sample + torch.randn_like(sample) * 0.2
            out_filp = th.max(operator.forward(sample_flip), 1)[1]
            torchvision.utils.save_image(label_as_rgb_visual(out_filp), os.path.join(out_path, 'low_res', fname))
        elif cfg['operator']['name'] == 'roomsegmentation':
            from roomsegmentation import visualize_result
            visualize_result(y_n, os.path.join(out_path, 'input', fname))
            visualize_result(operator.forward(sample), os.path.join(out_path, 'low_res', fname))
        else:
            torchvision.utils.save_image((y_n + 1.0) / 2.0, os.path.join(out_path, 'input', fname))
            torchvision.utils.save_image((operator.forward(sample) + 1.0) / 2.0, os.path.join(out_path, 'low_res', fname))
        torchvision.utils.save_image((ref_img + 1.0) / 2.0, os.path.join(out_path, 'label', fname))
        torchvision.utils.save_image((sample + 1.0) / 2.0, os.path.join(out_path, 'recon', fname))
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        distiller_path="",
        seed=42,
        ts="",
        cfg="",
        savedir="results/"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
