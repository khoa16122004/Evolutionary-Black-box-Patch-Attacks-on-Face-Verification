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
    dir_ = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\seed=22520691_arkiv_NSGAII_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal"
    
    adv_scores_log = []
    psnr_scores_log = []
    
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
            log = data_list["log"]

        # print("len log: ", len(log))
        # for i in range(0, len(log), 2):
        #     iter = log[i: i + 2]
        #     # print(len(iter[1]['adv_scores_log']))
        #     # print(iter[1]['adv_scores_log'])
            
        #     current_adv_scores = iter[0]['adv_scores_log'] + iter[1]['adv_scores_log']
        #     current_psnr_scores = iter[0]['psnr_scores_log'] + iter[1]['psnr_scores_log']
            
        #     adv_scores = []
        #     psnr_scores = []
        #     for j in range(len(current_adv_scores)):
        #         adv_scores.append(current_adv_scores[j].item())
        #         psnr_scores.append(current_psnr_scores[j].item())
        #     adv_scores_log.append(max(adv_scores))
        #     psnr_scores_log.append(max(psnr_scores))
        # print(adv_scores_log)
        
        for i in range(len(log)):
            iter = log[i]
            current_psnr_scores = iter['psnr_scores_log']
            current_adv_scores = iter['adv_scores_log']
            
            # print("adv_scores_log: ", current_adv_scores)
            # print("psnr_scores_log: ",current_psnr_scores)
            adv_scores = []

            psnr_scores = []
            for j in range(len(current_adv_scores)):
                adv_scores.extend(list(current_adv_scores[j].cpu()))
                psnr_scores.extend(list(current_psnr_scores[j].cpu()))
            adv_scores_log.append(max(adv_scores))
            psnr_scores_log.append(max(psnr_scores))
        
        break
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    axs[0].plot(range(len(adv_scores_log)), adv_scores_log, label="Adv Score", color="blue")
    axs[0].set_title("Adv Score over Iterations")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Adv Score")
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot(range(len(psnr_scores_log)), psnr_scores_log, label="PSNR Score", color="orange")
    axs[1].set_title("PSNR Score over Iterations")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("PSNR Score")
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

main()
