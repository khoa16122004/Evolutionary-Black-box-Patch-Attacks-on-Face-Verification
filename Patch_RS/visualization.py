import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2
from torchvision.utils import save_image
class Visualization:
    def __init__(self, i, data, outdir, last_result):
        # self.data = self._optimize_data(data)
        self.idx = i
        self.data = data
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.pickle_path = os.path.join(outdir, f"process_{self.idx}.pkl")
        self.last_result = last_result

    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
    def save_last_result(self):
        last_adv_image = self.last_result["x_adv"]
        last_adv_image = np.array(last_adv_image)
        save_image(torch.tensor(last_adv_image), os.path.join(self.outdir, f"last_adv_image_{self.idx}.png"))

    def plot_process(self):
        adv_score = [d["adv_score"] for d in self.data]
        psnr = [d["psnr"] for d in self.data]
        iterations = [d["iteration"] for d in self.data]
        best_adv_score = [d["best_adv_score"] for d in self.data]
        best_psnr = [d["best_psnr"] for d in self.data]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        ax1.plot(iterations, adv_score, label="Adversarial Score", color='tab:blue')
        ax1.set_title('Adversarial Score vs Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Adversarial Score')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(iterations, best_adv_score, label="Adversarial Score", color='tab:green')
        ax2.set_title('Best Adversarial Score vs Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Adversarial Score')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(iterations, psnr, label="PSNR", color='tab:red')
        ax3.set_title('PSNR vs Iteration')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('PSNR')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(iterations, best_psnr, label="PSNR", color='tab:purple')
        ax4.set_title('Best PSNR vs Iteration')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('PSNR')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"combined_plot_{self.idx}.png"))
        plt.close(fig)
        del adv_score, psnr, iterations, best_adv_score, best_psnr, fig, ax1, ax2, ax3, ax4

    def __call__(self):
        self.save_pickle(self.pickle_path)
        print(f"Saved process to {self.pickle_path}")
        self.plot_process()
        print(f"Plots saved to {self.outdir}")
        self.save_last_result()
        del self.data, self.last_result
        torch.cuda.empty_cache()
