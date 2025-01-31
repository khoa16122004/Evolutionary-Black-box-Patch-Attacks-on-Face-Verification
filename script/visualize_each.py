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

with open("22520601_rulebase_mean.txt", "r") as f1, open("22520691_rulebase_std.txt", "r") as f2:
    lines_mean = f1.readlines()
    lines_mean = [line.strip().split() for line in lines_mean]
    
    lines_std = f2.readlines()
    lines_std = [line.strip().split() for line in lines_std]
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

interval = 100
mean_adv_scores_log = [float(line[0]) for line in lines_mean][::interval]
mean_psnr_scores_log = [float(line[1]) for line in lines_mean][::interval]
std_adv_scores_log = [float(line[0]) for line in lines_std][::interval]
std_psnr_scores_log = [float(line[1]) for line in lines_std][::interval]
x = range(len(mean_adv_scores_log))
y_est_adv = np.array(mean_adv_scores_log)
y_err_adv = np.array(std_adv_scores_log)

y_est_psnr = np.array(mean_psnr_scores_log)
y_err_psnr = np.array(std_psnr_scores_log)



axs[0].plot(range(len(mean_adv_scores_log)), mean_adv_scores_log, label="Adv Score", color='red')
axs[1].plot(range(len(mean_psnr_scores_log)), mean_psnr_scores_log, label="PSNR Score", color='red')
axs[0].fill_between(x, y_est_adv - y_err_adv, y_est_adv + y_err_adv, color='red', alpha=0.2)
axs[1].fill_between(x, y_est_psnr - y_err_psnr, y_est_psnr + y_err_psnr, color='red', alpha=0.2)

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


