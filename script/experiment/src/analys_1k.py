import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# plt.rc('legend', fontsize=13)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=11)
plt.rc('axes', labelsize=13)
normal_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520691_normal_selected"
adaptive_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520691_adaptive_selected"
rulebased_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520691_rulebased_selected"
NSGAII_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520691_NSGAII_selected"

dir_dict = {
            "GA_normal": normal_dir, 
            "GA_adaptive": adaptive_dir,
            "GA_rules": rulebased_dir,
            "NSGAII": NSGAII_dir
            }



for key, value in dir_dict.items():
    adv_scores = [[] for _ in range(100)]
    psnr_scores = [[] for _ in range(100)]
    interval = 20
    count_1k = 0
    psnr_scores = 0
    for i, file_name in enumerate(os.listdir(dir_dict[key])):
        with open(os.path.join(dir_dict[key], file_name), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]

            for j in range(5000):
                if float(lines[j][0]) >= 0:
                    count_1k += 1
                    psnr_scores += float(lines[j][1])

                    break
                if j == 999:
                    psnr_scores += float(lines[j][1])

                # adv_scores[i].append(float(lines[j][0]))
                # psnr_scores[i].append(float(lines[j][1]))

    print(f"{key}: {count_1k}")
    print(f"{key}: {psnr_scores / 100}")