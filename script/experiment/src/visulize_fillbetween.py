import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rc('axes', labelsize=13)

normal_dir = r"normal\22520695_normal_selected"
adaptive_dir = r"adaptive\22520695_adaptive_selected"
rulebased_dir = r"rules\22520695_rules_selected"
NSGAII_dir = r"nsgaii\22520695_NSGAII_selected"
seed = "22520695"
interval = 500

dir_dict = {
    "GA w/ combined-objective": normal_dir,
    "GA w/ recontrucstion-bias": adaptive_dir,
    "GA w/ attack-bias": rulebased_dir,
    "GA w/ bi-objective": NSGAII_dir
}

dir_name = {
    "GA w/ combined-objective": 'normal',
    "GA w/ recontrucstion-bias": "adaptive",
    "GA w/ attack-bias": "rules",
    "GA w/ bi-objective": "nsgaii"
}



colors = {
    "GA w/ combined-objective": "black",
    "GA w/ recontrucstion-bias": "green",
    "GA w/ attack-bias": "blue",
    "GA w/ bi-objective": "red"
}

# Tạo folder lưu kết quả theo seed
save_folder = f"seed_{seed}"
os.makedirs(save_folder, exist_ok=True)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

for key, value in dir_dict.items():
    adv_scores = []
    psnr_scores = []
    
    for file_name in os.listdir(value):
        with open(os.path.join(value, file_name), "r") as f:
            lines = [line.strip().split() for line in f.readlines()]
            adv_scores.append([float(line[0]) for line in lines])
            psnr_scores.append([float(line[1]) for line in lines])

    adv_scores = np.array(adv_scores)
    psnr_scores = np.array(psnr_scores)

    mean_adv_scores = np.mean(adv_scores, axis=0)
    mean_psnr_scores = np.mean(psnr_scores, axis=0)
    
    std_adv_scores = np.std(adv_scores, axis=0)
    std_psnr_scores = np.std(psnr_scores, axis=0)

    adv_lower = mean_adv_scores - std_adv_scores
    adv_upper = mean_adv_scores + std_adv_scores
    psnr_lower = mean_psnr_scores - std_psnr_scores
    psnr_upper = mean_psnr_scores + std_psnr_scores

    x = np.arange(0, len(mean_adv_scores), interval)
    axs[0].plot(x, mean_adv_scores[::interval], color=colors[key], label=f"{key}")
    axs[1].plot(x, mean_psnr_scores[::interval], color=colors[key], label=f"{key}")
    axs[0].fill_between(x, adv_lower[::interval], adv_upper[::interval], color=colors[key], alpha=0.1)
    axs[1].fill_between(x, psnr_lower[::interval], psnr_upper[::interval], color=colors[key], alpha=0.1)

    # Lưu mean_adv_scores và mean_psnr_scores vào thư mục seed_{seed}
    # np.savetxt(os.path.join(save_folder, f"{dir_name[key]}_adv.txt"), mean_adv_scores, fmt="%.6f")
    # np.savetxt(os.path.join(save_folder, f"{dir_name[key]}_psnr.txt"), mean_psnr_scores, fmt="%.6f")

axs[0].set_title("Adversarial Score over Iterations")
axs[0].set_ylabel("Adversarial Score")
axs[0].grid(True)
axs[0].legend()

axs[1].set_title("PSNR Score over Iterations")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("PSNR Score")
axs[1].grid(True)
axs[1].legend()

fig.tight_layout(pad=2.0)
# plt.savefig(os.path.join(save_folder, "mean_visualize.pdf"), format="pdf", bbox_inches="tight")
plt.show()
