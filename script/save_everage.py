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

def take_log(log, type_='GA'):
    if type_ == "GA":
        adv_scores_log = []
        psnr_scores_log = []
        for i in range(0, len(log), 2):
            iter = log[i: i + 2]

            current_adv_scores = iter[0]['adv_scores_log'] + iter[1]['adv_scores_log']
            current_psnr_scores = iter[0]['psnr_scores_log'] + iter[1]['psnr_scores_log']

            adv_scores = [score.item() for score in current_adv_scores]
            psnr_scores = [score.item() for score in current_psnr_scores]

            adv_scores_log.append(max(adv_scores))
            psnr_scores_log.append(max(psnr_scores))

        return adv_scores_log, psnr_scores_log

    else:
        best_PNSR_adv_scores = []
        best_PNSR_psnr_scores = []
        best_ADV_adv_scores = []
        best_ADV_psnr_scores = []
        for i in range(len(log)):
            iter = log[i]
            current_psnr_scores = iter['psnr_scores_log']
            current_adv_scores = iter['adv_scores_log']

            psnr_scores = [score.cpu().item() for scores in current_psnr_scores for score in scores]
            adv_scores = [score.cpu().item() for scores in current_adv_scores for score in scores]

            best_PNSR_index = argmax(psnr_scores)
            best_ADV_index = argmax(adv_scores)

            best_PNSR_adv_scores.append(current_adv_scores[best_PNSR_index])
            best_PNSR_psnr_scores.append(current_psnr_scores[best_PNSR_index])
            best_ADV_adv_scores.append(current_adv_scores[best_ADV_index])
            best_ADV_psnr_scores.append(current_psnr_scores[best_ADV_index])

        return best_PNSR_adv_scores, best_PNSR_psnr_scores, best_ADV_adv_scores, best_ADV_psnr_scores

def main():
    pkl_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\experiment\seed=22520691_arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle"
    output_file_mean = "22520601_rulebase_mean.txt"
    output_file_std = "22520691_rulebase_std.txt"
    
    interval = 1
    num_samples = 200
    num_intervals = 10000 // interval  

    all_adv_scores = np.zeros((num_samples, num_intervals))
    all_psnr_scores = np.zeros((num_samples, num_intervals))
    
    for i in tqdm(range(100)):
        input_pkl = os.path.join(pkl_dir, f"{i}.pkl")
        with open(input_pkl, "rb") as f:
            log = pkl.load(f)['log']

        adv_scores_log, psnr_scores_log = take_log(log, type_="GA")

        for idx, j in enumerate(range(0, 10000, interval)):
            all_adv_scores[i, idx] = adv_scores_log[j]
            all_psnr_scores[i, idx] = psnr_scores_log[j]

    mean_adv_scores = np.mean(all_adv_scores, axis=0)
    std_adv_scores = np.std(all_adv_scores, axis=0)

    mean_psnr_scores = np.mean(all_psnr_scores, axis=0)
    std_psnr_scores = np.std(all_psnr_scores, axis=0)

    with open(output_file_mean, "w") as f_mean, open(output_file_std, "w") as f_std:
        for mean_adv, std_adv, mean_psnr, std_psnr in zip(mean_adv_scores, std_adv_scores, mean_psnr_scores, std_psnr_scores):
            f_mean.write(f"{mean_adv} {mean_psnr}\n")
            f_std.write(f"{std_adv} {std_psnr}\n")

if __name__ == "__main__":
    main()