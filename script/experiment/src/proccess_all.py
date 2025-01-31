import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rc('axes', labelsize=13)

normal_dir = r"normal"
adaptive_dir = r"adaptive"
rulebased_dir = r"rules"
NSGAII_dir = r"nsgaii"
seed = "22520695"
interval = 10

dir_dict = {
    "Combined fitness": normal_dir,
    "Reconstruction-bias": adaptive_dir,
    "Attack-bias": rulebased_dir,
    "NSGA-II": NSGAII_dir
}

dir_name = {
    "Combined fitness": 'normal',
    "Reconstruction-bias": "adaptive",
    "Attack-bias": "rules",
    "NSGA-II": "nsgaii"
}



colors = {
    "Combined fitness": "black",
    "Reconstruction-bias": "green",
    "Attack-bias": "blue",
    "NSGA-II": "red"
}



fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].set_xlim(0, 10000)
axs[1].set_xlim(0, 10000)
axs[0].tick_params(axis='both', labelsize=14)
axs[1].tick_params(axis='both', labelsize=14)
for method, value in dir_dict.items():
    # value: normal
    adv_scores_all = []
    psnr_scores_all = []
    for seed_dir in os.listdir(value):
        seed_folder = os.path.join(value, seed_dir)
        adv_scores = [] # 100 x 10000
        psnr_scores = [] # 100 x 10000
        
        for file_name in os.listdir(seed_folder):
            with open(os.path.join(seed_folder, file_name), "r") as f:
                lines = [line.strip().split() for line in f.readlines()]
                adv_scores.append([float(line[0]) for line in lines])
                psnr_scores.append([float(line[1]) for line in lines])

        adv_scores_all.extend(adv_scores)
        psnr_scores_all.extend(psnr_scores)
        
    adv_scores_all = np.array(adv_scores_all)
    psnr_scores_all = np.array(psnr_scores_all)
    
    mean_adv_scores = np.mean(adv_scores_all, axis=0)
    mean_psnr_scores = np.mean(psnr_scores_all, axis=0)
    
    std_adv_scores = np.std(adv_scores_all, axis=0)
    std_psnr_scores = np.std(psnr_scores_all, axis=0)

    adv_lower = mean_adv_scores - std_adv_scores
    adv_upper = mean_adv_scores + std_adv_scores
    psnr_lower = mean_psnr_scores - std_psnr_scores
    psnr_upper = mean_psnr_scores + std_psnr_scores

    x = np.arange(0, len(mean_adv_scores), interval)
    axs[0].plot(x, mean_adv_scores[::interval], color=colors[method], label=f"{method}")
    axs[1].plot(x, mean_psnr_scores[::interval], color=colors[method], label=f"{method}")
    axs[0].fill_between(x, adv_lower[::interval], adv_upper[::interval], color=colors[method], alpha=0.1)
    axs[1].fill_between(x, psnr_lower[::interval], psnr_upper[::interval], color=colors[method], alpha=0.1)
        
        
    # adv_scores_all: 5 x 100 x 10000
    # trải ra thành 500 x 10000
    # lưu lại adv_scores_all và psnr_scores_all với mảng numpy
    # break
        # 
    # save name method,     
axs[0].set_title("Adversarial Score over Iterations")
axs[0].set_ylabel("Adversarial Score")
axs[0].grid(True)
axs[0].legend(fontsize=14)

axs[1].set_title("PSNR Score over Iterations")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("PSNR Score")
axs[1].grid(True)
axs[1].legend(fontsize=14)
fig.tight_layout(pad=2.0)
plt.savefig('iterations_visualize.pdf', format="pdf", bbox_inches="tight")
plt.show()



