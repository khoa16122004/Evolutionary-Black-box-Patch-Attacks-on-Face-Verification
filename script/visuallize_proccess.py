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

# vỡi mỗi tấm:
# -GA adaptive, -GA normal
## với mỗi iter lấy ra 1 indvidual với max combined/adaptive fitness: psnr, adv
# - NSGA
## lấy ra thằng theo rulebased: psnr, adv
# -> 100 x 10000

# với mỗi iter lấy ra : std, mean



def argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])

def selection(Population):
    valid_population = [ind for ind in Population if ind.adv_score.item() >= 0]

    if valid_population:
        return max(valid_population, key=lambda ind: ind.psnr_score.item())
    else:
        return max(Population, key=lambda ind: ind.adv_score.item())
# NSGA II nét liền và đức
def take_log(log, type_='GA'):

    if type_.startswith("GA"):
        adv_scores_log = []
        psnr_scores_log = []
        
        for i in range(0, len(log), 2):
            iter = log[i: i + 2]
            
            current_adv_scores = iter[0]['adv_scores_log'] + iter[1]['adv_scores_log']
            current_psnr_scores = iter[0]['psnr_scores_log'] + iter[1]['psnr_scores_log']
            current_combined = []
            if type_ == "GA_normal":
                current_combined = [
                    0.5 * current_adv_scores[j] + 0.5 * current_psnr_scores[j] 
                    for j in range(len(current_adv_scores))
                ]
            elif type_ == "GA_adaptive":
                current_combined = [
                    0.5 * current_psnr_scores[j] if current_adv_scores[j] >= 0 
                    else 0.5 * current_adv_scores[j] + 0.5 * current_psnr_scores[j] 
                    for j in range(len(current_adv_scores))
                ]
            else:
                indexs_postive = []
                for k in range(len(current_adv_scores)):
                    if current_adv_scores[k] >= 0:
                        indexs_postive.append(k)
                if len(indexs_postive) > 0:
                    psnr_postives = [current_psnr_scores[i] for i in indexs_postive]
                    adv_postives = [current_adv_scores[i] for i in indexs_postive]
                    best_idx = argmax(psnr_postives)
                    
                    adv_scores_log.append(psnr_postives[best_idx])
                    psnr_scores_log.append(adv_postives[best_idx])
                else:
                    best_adv_index = argmax(current_adv_scores)
                    adv_scores_log.append(current_adv_scores[best_adv_index])
                    psnr_scores_log.append(current_psnr_scores[best_adv_index])
                
                continue
                    

            
            adv_scores = []
            psnr_scores = []
            for j in range(len(current_adv_scores)):
                best_combined_index = argmax(current_combined)
                adv_scores.append(current_adv_scores[best_combined_index].item())
                psnr_scores.append(current_psnr_scores[best_combined_index].item())
            adv_scores_log.append(max(adv_scores))
            psnr_scores_log.append(max(psnr_scores))
            
        return adv_scores_log, psnr_scores_log

    else:
        best_PNSR_adv_scores = [] # the adv scores of the best PNSR scores
        best_PNSR_psnr_scores = [] # the psnr scores of the best PNSR scores
        best_ADV_adv_scores = [] # the adv scores of best ADV scores
        best_ADV_psnr_scores = [] # the psnr scores of the best ADV scores
        for i in range(len(log)):
            iter = log[i]
            current_psnr_scores = iter['psnr_scores_log']
            current_adv_scores = iter['adv_scores_log']
            
            psnr_scores = []
            adv_scores = []
            for j in range(len(current_adv_scores)):
                adv_scores.extend(list(current_adv_scores[j].cpu()))
                psnr_scores.extend(list(current_psnr_scores[j].cpu()))
                
            # take the index of the best PNSR and ADV
            best_PNSR_index = argmax(psnr_scores)
            best_ADV_index = argmax(adv_scores)
            
            best_PNSR_adv_score = adv_scores[best_PNSR_index]
            best_PNSR_psnr_score = psnr_scores[best_PNSR_index]
            
            best_ADV_adv_score = adv_scores[best_ADV_index]
            best_ADV_psnr_score = psnr_scores[best_ADV_index]
            
            best_PNSR_adv_scores.append(best_PNSR_adv_score)
            best_PNSR_psnr_scores.append(best_PNSR_psnr_score)
            best_ADV_adv_scores.append(best_ADV_adv_score)
            best_ADV_psnr_scores.append(best_ADV_psnr_score)
            
        return best_PNSR_adv_scores, best_PNSR_psnr_scores, best_ADV_adv_scores, best_ADV_psnr_scores
    

def main():
    acc = 0
    psnr = 0
    interval = 100
    toTensor = transforms.ToTensor()
    MODEL = get_model("restnet_vggface")
    DATA = LFW(IMG_DIR=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\lfw_crop_margin_5",
                MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
                PAIR_PATH=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\pairs.txt",
                transform=None)
    i = 0
    ######
    dir_GA_normal = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\seed=22520691_arkiv_GA_niter=10000_label=0_reconsw=0.5_attackw=0.5_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle\0.pkl"
    dir_GA_adaptive = r"D:\Path-Recontruction-with-Evolution-Strategy\POPOP_location\arkiv_GA_niter=10000_label=0_reconsw=0.5_attackw=0.5_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=adaptive\pickle\0.pkl"
    dir_GA_rule_based = r"D:\Path-Recontruction-with-Evolution-Strategy\POPOP_location\arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle\0.pkl"
    dir_NSGAII = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\seed=22520691_arkiv_NSGAII_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle\0.pkl"

    dir_dict = {
                "GA_normal": dir_GA_normal, 
                "GA_adaptive": dir_GA_adaptive,
                "GA_rules": dir_GA_rule_based,
                # "NSGAII_best_PSNR": dir_NSGAII,
                # "NSGAII_Best_ADV": ""
                }

    data_dict = {}

    for key, value in tqdm(dir_dict.items()):
        if key.startswith("GA"):
            data_dict[key] = take_log(pkl.load(open(value, "rb"))['log'], key)
        else:
            print("NSGAII")
            best_PNSR_adv_scores, best_PNSR_psnr_scores, best_ADV_adv_scores, best_ADV_psnr_scores = take_log(pkl.load(open(value, "rb"))['log'], "NSGAII")
            data_dict["NSGAII_best_PSNR"] = (best_PNSR_adv_scores, best_PNSR_psnr_scores)
            data_dict["NSGAII_best_ADV"] = (best_ADV_adv_scores, best_ADV_psnr_scores)
            break

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    colors = {
                "GA_normal": "blue", 
              "GA_adaptive": "green", 
              "GA_rules": "orange",
            #   "NSGAII_best_PSNR": "red",
            #   "NSGAII_best_ADV": "black",
              }
    
    markers = {
                "GA_normal": "o", 
              "GA_adaptive": "s", 
              "GA_rules": "D",
            #   "NSGAII_best_PSNR": "p",
            #   "NSGAII_best_ADV": "X",
              }
    
    interval = 200
    for key, (adv_scores_log, psnr_scores_log) in data_dict.items():
        axs[0].plot(range(0, len(adv_scores_log), interval), adv_scores_log[::interval], label=f"{key} Adv Score", color=colors[key], marker=markers[key])
        axs[1].plot(range(0, len(psnr_scores_log), interval), psnr_scores_log[::interval], label=f"{key} PSNR Score", color=colors[key], marker=markers[key])



    axs[0].set_title("Adv Score over Iterations")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Adv Score")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("PSNR Score over Iterations")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("PSNR Score")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

main()
