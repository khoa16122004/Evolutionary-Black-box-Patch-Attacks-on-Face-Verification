import numpy as np  
import torch
import pickle
from tqdm import tqdm
from PIL import Image
import os

class PatchBruteForce:
    def __init__(self, i, loss, patch_size, outdir):
        self.idx = i
        self.loss = loss
        self.patch_size = patch_size
        self.process = []
        self.outdir = os.path.join(outdir, f"image-{self.idx}")
        os.makedirs(self.outdir, exist_ok=True)

    def apply_patch(self, x, patch_loc, patch_delta):
        x_perturbed = x.clone()
        h_start, w_start = patch_loc
        h_end = h_start + self.patch_size
        w_end = w_start + self.patch_size

        if patch_delta.shape != (x.shape[0], self.patch_size, self.patch_size):
            raise ValueError(
                f"Patch delta shape {patch_delta.shape} does not match expected shape {(x.shape[0], self.patch_size, self.patch_size)}"
            )

        x_perturbed[:, h_start:h_end, w_start:w_end] += patch_delta
        return torch.clamp(x_perturbed, 0, 1)     
    def attack(self, x):
        img_shape = x.shape
        # print("IMG: ", img_shape)
        # return 
        patch_loc = (0, 0)
        patch_delta = np.random.normal(0, 0.5, (img_shape[0], self.patch_size, self.patch_size))
        x_adv = self.apply_patch(x, patch_loc, patch_delta)
        idx = 0
        total_iterations = (img_shape[1] - self.patch_size + 1) * (img_shape[2] - self.patch_size + 1)
        suc_cnt = 0

        best_adv_score = float('-inf')
        best_psnr = float('-inf')
        best_psnr_adv, best_adv = None, None

        # debug phase
        # adv_score_success = []
        with tqdm(total=total_iterations, desc="Processing patches") as pbar:
            for h_start in range(0, img_shape[1] - self.patch_size + 1):
                for w_start in range(0, img_shape[2] - self.patch_size + 1):
                    patch_loc = (h_start, w_start)
                    x_candidate = self.apply_patch(x, patch_loc, patch_delta)
                    psnr, adv_score, success = self.loss(x_candidate)
                    if success:
                        if psnr > best_psnr:
                            best_psnr = psnr
                            best_psnr_adv = {
                                "adv_score": adv_score.cpu().item(),
                                "success": success.cpu().item(),
                                "patch_loc": np.array(patch_loc),
                                "patch_delta": patch_delta,
                                "psnr": psnr.cpu().item(),
                            }
                        x_adv = x_candidate
                        self.process.append({
                            "adv_score": adv_score.cpu().item(),
                            "success": success.cpu().item(),
                            "patch_loc": np.array(patch_loc),
                            "patch_delta": patch_delta,
                            "psnr": psnr.cpu().item(),
                        })
                        if adv_score > best_adv_score:
                            self.save_image(x_adv, os.path.join(self.outdir, f"{self.idx}_{h_start}-{w_start}.png"))
                        suc_cnt += 1    
                        # adv_score_success.append(adv_score.cpu().item())
                    if adv_score > best_adv_score:
                        best_adv_score = adv_score
                        best_adv  = {
                            "adv_score": adv_score.cpu().item(),
                            "success": success.cpu().item(),
                            "patch_loc": np.array(patch_loc),
                            "patch_delta": patch_delta,
                            "psnr": psnr.cpu().item(),
                        }
                    idx += 1
                    pbar.update(1)
        # os.makedirs(self.outdir, exist_ok=True)
        print(f"\nSuccess rate: {suc_cnt}/{total_iterations}\n")
        # print(f"\nDEBUG: {adv_score_success}\n")
        self.save_pickle(os.path.join(self.outdir, f"process_{self.idx}.pkl"))
        self.save_best(x, best_psnr_adv, best_adv, self.outdir)
        return patch_delta
    
    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.process, f)
    def save_best(self, x, best_psnr_adv, best_adv, outdir):
        os.makedirs(outdir, exist_ok=True)
        if best_psnr_adv:
            best_psnr_adv_img = self.apply_patch(x, tuple(best_psnr_adv["patch_loc"]), best_psnr_adv["patch_delta"])
            self.save_image(best_psnr_adv_img, os.path.join(outdir, f"best_psnr_adv_{self.idx}.png"))
        best_adv_img = self.apply_patch(x, tuple(best_adv["patch_loc"]), best_adv["patch_delta"])
        self.save_image(best_adv_img, os.path.join(outdir, f"best_adv_{self.idx}.png"))
        if best_psnr_adv:
            best_psnr_adv_img = self.apply_patch(x, tuple(best_psnr_adv["patch_loc"]), best_psnr_adv["patch_delta"])
            self.save_image(best_psnr_adv_img, os.path.join(outdir, f"best_psnr_adv_{self.idx}.png"))
            with open(os.path.join(outdir, f"best_adv_{self.idx}.txt"), 'a') as f:
                f.write(f"{self.idx}: {best_psnr_adv['adv_score']} {best_psnr_adv['psnr']}, {best_adv['adv_score']} {best_adv['psnr']}\n")
            return
        with open(os.path.join(outdir, f"best_adv_{self.idx}.txt"), 'a') as f:
            f.write(f"{self.idx}: {best_adv['adv_score']} {best_adv['psnr']}\n")
    def save_image(self, x, path):
        x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        x = (x * 255).astype(np.uint8)
        Image.fromarray(x).save(path)