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
                if len(indexs_postive > 0):
                    psnr_postives = psnr_postives[indexs_postive]
                    adv_postives = adv_scores[indexs_postive]
                    best_idx = argmax(psnr_postives)
                    
                    adv_scores_log.append(psnr_postives[best_idx])
                    psnr_scores_log.append(adv_postives[best_idx])
                else:
                    best_adv_index = argmax(adv_scores_log)
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
    pickle_dir = ""
    type_ = "GA"
    
    
    
    
main()
