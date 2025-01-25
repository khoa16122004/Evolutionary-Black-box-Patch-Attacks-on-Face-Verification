import random
import numpy as np
from visualization import Visualization
from tqdm import tqdm
import torch

class SparseRSAttack:
    def __init__(self, idx, loss, patch_size, update_loc_period, max_query, reconstruct=False, outdir="results"):
        self.loss = loss
        self.idx = idx
        self.patch_size = patch_size
        self.update_loc_period = update_loc_period
        self.max_query = max_query
        self.reconstruct = reconstruct
        self.outdir = outdir
        self.process = []
        self.last_result = None

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

    def random_patch(self, img_shape, original_patch_delta=None, max_l2_diff=0.1):
        h_start = random.randint(0, img_shape[1] - self.patch_size)
        w_start = random.randint(0, img_shape[2] - self.patch_size)
        patch_loc = (h_start, w_start)

        if original_patch_delta is not None:
            patch_delta = original_patch_delta + np.random.normal(0, max_l2_diff, original_patch_delta.shape)
        else:
            patch_delta = np.random.uniform(-1, 1, (img_shape[0], self.patch_size, self.patch_size))

        return patch_loc, torch.clamp(torch.tensor(patch_delta, dtype=torch.float32), -1, 1)

    def attack(self, x):
        img_shape = x.shape

        patch_loc, patch_delta = self.random_patch(img_shape)

        x_adv = self.apply_patch(x, patch_loc, patch_delta)
        best_psnr, best_adv_score, success = self.loss(x_adv)

        for i in tqdm(range(self.max_query), desc="Attack Progress"):
            if success:
                if self.reconstruct:
                    _, patch_delta_new = self.random_patch(img_shape, original_patch_delta=patch_delta, max_l2_diff=0.2)
                    x_candidate = self.apply_patch(x, patch_loc, patch_delta_new)
                    psnr, adv_score, success = self.loss(x_candidate)
                    if psnr > best_psnr and success:
                        best_psnr = psnr
                        patch_delta = patch_delta_new
                        x_adv = x_candidate
                        self.process.append({
                            "iteration": i,
                            "psnr": psnr.cpu().item(),
                            "best_psnr": best_psnr.cpu().item(),
                            "adv_score": adv_score.cpu().item(),
                            "best_adv_score": best_adv_score.cpu().item(),
                            "success": success.cpu().item(),
                            "patch_loc": np.array(patch_loc),
                            "patch_delta": patch_delta.cpu().numpy()
                        })
                        continue
                else:
                    print(f"\nAttack success without reconstruction at iteration ", i-1)
                    break
            if i % self.update_loc_period == 0:
                patch_loc, _ = self.random_patch(img_shape)

            _, patch_delta_new = self.random_patch(img_shape, original_patch_delta=patch_delta, max_l2_diff=0.1)
            x_candidate = self.apply_patch(x, patch_loc, patch_delta_new)

            psnr, adv_score, success = self.loss(x_candidate)

            if adv_score > best_adv_score:
                best_adv_score = adv_score
                patch_delta = patch_delta_new

            x_adv = self.apply_patch(x, patch_loc, patch_delta)
            
            # print("PSNR", psnr)
            self.process.append({
                "iteration": i,
                "psnr": psnr.cpu().item(),
                "best_psnr": best_psnr.cpu().item(),
                "adv_score": adv_score.cpu().item(),
                "best_adv_score": best_adv_score.cpu().item(),
                "success": success.cpu().item(),
                "patch_loc": np.array(patch_loc),
                "patch_delta": patch_delta.cpu().numpy(),
            })
        self.last_result = {
            "best_psnr": best_psnr.cpu().item(),
            "best_adv_score": best_adv_score.cpu().item(),
            "success": success.cpu().item(),
            "x_adv": x_adv.cpu().detach().numpy().tolist()
        }
        visual = Visualization(self.idx, self.process, self.outdir, self.last_result)
        visual()
        return self.last_result
