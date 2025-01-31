import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy.spatial import ConvexHull
import tkinter as tk
from PIL import Image, ImageTk
import pickle as pkl
from get_architech import get_model
from dataset import LFW
from fitness import Fitness
from torchvision import transforms
import os
import cv2 as cv
from torchvision.utils import save_image
from tqdm import tqdm

def selection(Population):
    valid_population = [ind for ind in Population if ind.adv_score.item() >= 0]
    
    if valid_population:
        return max(valid_population, key=lambda ind: ind.psnr_score.item())
    else:
        return max(Population, key=lambda ind: ind.adv_score.item())
    
def main():
    acc = 0
    psnr = 0
    toTensor = transforms.ToTensor()
    MODEL = get_model("restnet_vggface")
    DATA = LFW(IMG_DIR=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\lfw_crop_margin_5",
                MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
                PAIR_PATH=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\pairs.txt",
                transform=None)
    ######
    dir_ = r"D:\Path-Recontruction-with-Evolution-Strategy\POPOP_location\arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal"
    for i in tqdm(range(100)):
        pickle_file = os.path.join(dir_, "pickle", f"{i}.pkl")
        greedy_folder = os.path.join(dir_, "greedy")
        os.makedirs(greedy_folder, exist_ok=True)
        
    
    
        img1 = DATA[i][0].resize((160, 160))
        img1 = toTensor(img1)

        fitness = Fitness(patch_size=20,
                                img1=img1,
                                img2=None,
                                model=MODEL,  # Use the correct model
                                label=0,
                                recons_w=None,
                                attack_w=None,
                                fitness_type=None)
        
        
        with open(pickle_file, "rb") as f:
            data_list = pkl.load(f)
            Population = data_list["Population"]

        final_ind = selection(Population)
        adv_img = fitness.apply_patch_to_image(final_ind.patch, final_ind.location)
        if final_ind.adv_score.item() >= 0:
            acc += 1
        save_image(adv_img, os.path.join(greedy_folder, f"{i}.png"))
        # print("Adv_score: ", final_ind.adv_score.item())
        # print("PSNR_score: ", final_ind.psnr_score.item())
        psnr += final_ind.psnr_score.item()
    print("PNSR_score: ", psnr / 100)
    print("Accuracy: ", acc / 100)
main()
    
