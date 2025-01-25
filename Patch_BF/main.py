import argparse
from .get_architech import get_model
from .loss import LossBF
from .algorithm import PatchBruteForce
from torchvision import transforms
import torch
from PIL import Image
from .dataset import LFW
import random
import pickle
from tqdm import tqdm
import numpy as np
import os
def parse_args():
    parser = argparse.ArgumentParser(description="Patch BF for Image Patch Manipulation")
    parser.add_argument('--patch_size', type=int, default=20, help="Size of the patch")
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--indir', type=str, default='\kaggle\input\lfw_dataset')
    parser.add_argument('--pretrained_dir', type=str, default='\kaggle\input\face-verification')
    # parser.add_argument('--num_imgs', type=int, default=100)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)
    parser.add_argument('--seed', type=int, default=22520692)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL = get_model("restnet_vggface", args.pretrained_dir)
    DATA = LFW(IMG_DIR=os.path.join(args.indir, "lfw_crop_margin_5", "lfw_crop_margin_5"),
               MASK_DIR=r"D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_lips_mask", 
               PAIR_PATH=os.path.join(args.indir, "pairs.txt"),
               transform=None)
    toTensor = transforms.ToTensor()
    all_res = []
    # print("RECON", args.reconstruct)
    for i in tqdm(range(args.start_idx, args.end_idx), desc="Image Processing"):
        random.seed(args.seed)
        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)
        loss = LossBF(img1_torch, img2_torch, MODEL)
        attack = PatchBruteForce(i, loss, args.patch_size, args.outdir)
        res = attack.attack(img1_torch)
        all_res.append(res)
    with open(f"{args.outdir}/all_res.pkl", "wb") as f:
        pickle.dump(all_res, f)



