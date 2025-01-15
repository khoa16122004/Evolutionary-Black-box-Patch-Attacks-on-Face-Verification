import argparse
from get_architech import get_model
from loss import LossRS
from algorithm import SparseRSAttack
from torchvision import transforms
import torch
from PIL import Image
from dataset import LFW
import random
import pickle
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Patch RS for Image Patch Manipulation")
    parser.add_argument('--patch_size', type=int, default=16, help="Size of the patch")
    parser.add_argument('--n_iter', type=int, default=100, help="Number of iterations")
    parser.add_argument('--update_location_iterval', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--reconstruct', type=int, default=1)
    parser.add_argument('--indir', type=str, default='\kaggle\input\lfw_dataset')
    parser.add_argument('--pretrained_dir', type=str, default='\kaggle\input\face-verification')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL = get_model("restnet_vggface", args.pretrained_dir)

    DATA = LFW(IMG_DIR=f"{args.indir}\lfw_crop_margin_5\lfw_crop_margin_5",
               MASK_DIR=r"D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_lips_mask", 
               PAIR_PATH=f"{args.indir}\pairs.txt",
               transform=None)
    toTensor = transforms.ToTensor()
    all_res = []
    print("RECON", args.reconstruct)
    for i in tqdm(range(0, 1), desc="Image Processing"):
        random.seed(22520691)
        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)
        loss = LossRS(img1_torch, img2_torch, MODEL)
        attack = SparseRSAttack(i, loss, args.patch_size, args.update_location_iterval, args.n_iter, args.reconstruct, args.outdir)
        res = attack.attack(img1_torch)
        all_res.append(res)
    with open(f"{args.outdir}/all_res.pkl", "wb") as f:
        pickle.dump(all_res, f)



