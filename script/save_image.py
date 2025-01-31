import os
from tqdm import tqdm
import pickle as pkl
import argparse
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
def argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])

def ruled_selection(population, iter_adv_scores, iter_psnr_scores):
    success_indexs = []
    for k in range(len(iter_adv_scores)):
        if iter_adv_scores[k] >= 0:
            success_indexs.append(k)

    if len(success_indexs) > 0: # if exist successfully
        iter_success_psnr_scores = [iter_psnr_scores[i] for i in success_indexs] # pnsr of success
        iter_success_adv_scores = [iter_adv_scores[i] for i in success_indexs] # adv of success
        iter_success_indvidual = [population[i] for i in success_indexs] #
        
        
        best_psnr_iter_success = argmax(iter_success_psnr_scores)
        return iter_success_indvidual[best_psnr_iter_success]
    
    else:
        best_adv_iter = argmax(iter_adv_scores)
        return population[best_adv_iter]
def combined_selection(population, iter_adv_scores, iter_psnr_scores, algorithm):
    if algorithm == "GA_normal":
        current_combined_fitnesses = [
            0.5 * iter_adv_scores[j] + 0.5 * iter_psnr_scores[j] 
            for j in range(len(iter_adv_scores))
        ]

    elif algorithm == "GA_adaptive":
        current_combined_fitnesses = [
                    0.5 * iter_psnr_scores[j] if iter_adv_scores[j] >= 0 
                    else 0.5 * iter_adv_scores[j] + 0.5 * iter_psnr_scores[j] 
                    for j in range(len(iter_adv_scores))
                ]
    best_combined_fitnesse_index = argmax(current_combined_fitnesses)
    return population[best_combined_fitnesse_index]
        
    
def take_data(pkl_file, algorithm):
    # return log and return final Population
    population = pkl_file['Population']
    
    final_adv_scores = [ind.adv_score for ind in population]
    final_psnr_scores = [ind.psnr_score for ind in population]
    indv_rule = ruled_selection(population, final_adv_scores, final_psnr_scores)
    indv_combined = None
    if algorithm == 'GA_normal' or algorithm == 'GA_adaptive':
        indv_combined = combined_selection(population, final_adv_scores, final_psnr_scores, algorithm)
    
    return indv_rule, indv_combined
    
def load_file(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pkl.load(f)
    
def main(args):
    output_img_dir = os.path.join(args.output_seleted_dir, "img")
    os.makedirs(output_img_dir, exist_ok=True) 
    
    toTensor = transforms.ToTensor()
    MODEL = get_model("restnet_vggface")
    DATA = LFW(IMG_DIR=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\lfw_crop_margin_5",
                MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
                PAIR_PATH=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\pairs.txt",
                transform=None)
       
    for i in tqdm(range(len(os.listdir(args.pkl_dir)))):
        
        img1, img2 = DATA[i][0].resize((160, 160)), DATA[i][1].resize((160, 160))
        img1, img2 = toTensor(img1), toTensor(img2)

        fitness = Fitness(patch_size=20,
                                img1=img1,
                                img2=img2,
                                model=MODEL,
                                label=0,
                                recons_w=None,
                                attack_w=None,
                                fitness_type=None)
        
        
        pkl_path = os.path.join(args.pkl_dir, f"{i}.pkl")

        print("Loading file: ", pkl_path)
        try:
            pkl_file = load_file(pkl_path)
        except:
            continue
        indv_rule, indv_combined = take_data(pkl_file, args.algorithm)
        indv_rule_image = fitness.apply_patch_to_image(indv_rule.patch, indv_rule.location)
        save_image(indv_rule_image, os.path.join(output_img_dir, f"{i}.png"))    

        
        if indv_combined:
            indv_combined_image = fitness.apply_patch_to_image(indv_combined.patch, indv_combined.location)
            save_image(indv_combined_image, os.path.join(output_img_dir, f"{i}.png"))    

        
            
        # break

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_dir", type=str, required=True)
    parser.add_argument("--output_seleted_dir", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    args = parser.parse_args()
    main(args)
# python process_result.py     
# --pkl_dir D:\Path-Recontruction-with-Evolution-Strategy\experiment\seed=22520691_arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle
# --output_selected_dir 22520691_adaptive_selected
# --output_final_selected_dir 22520691_adaptive_final_selected
# --algorithm GA_adaptive


# python process_result.py --pkl_dir D:\Path-Recontruction-with-Evolution-Strategy\experiment\seed=22520691_arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle --output_seleted_dir 22520691_rulebased_selected --output_final_seleted_dir 22520691_adaptive_rulebased_selected --algorithm GA_rulebased
