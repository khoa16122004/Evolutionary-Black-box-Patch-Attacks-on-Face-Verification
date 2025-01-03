import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
from PIL import *
import torch

def l2(adv_patch, orig_patch, device='cuda'):
    assert adv_patch.shape == orig_patch.shape
    if not isinstance(adv_patch, torch.Tensor):
        adv_patch = torch.tensor(adv_patch, device=device)
    if not isinstance(orig_patch, torch.Tensor):
        orig_patch = torch.tensor(orig_patch, device=adv_patch.device, dtype=adv_patch.dtype)
    return torch.sum((adv_patch - orig_patch) ** 2)


def sh_selection(n_queries, it):
    """ schedule to decrease the parameter p 
        giới hạn lại chiều cao, nơi mà có thể tấn công (bằng 75% chiều cao của ảnh * factor điều chỉnh phạm vi dựa trên iteration)
        => iter càng tăng (càng gần với n queries, càng gần với kết thúc)
        -> phạm vi tấn công càng hẹp lại/càng gần với patch thành công trước đó
    """
    t = max((float(n_queries - it) / n_queries - .0) ** 1., 0) * .75

    return t


def update_location(loc_new, h_i, h, s):
    loc_new += np.random.randint(low=-h_i, high=h_i + 1, size=(2,))
    loc_new = np.clip(loc_new, 0, h - s)
    return loc_new


def render(x, w, device='cuda'):
    # Convert initial phenotype to CUDA tensor
    phenotype = torch.ones((w, w, 3)).to(device)
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    
    # Move x to CUDA if it's not already there
    x_cuda = torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x.to(device)
    
    for row in x_cuda:
        overlay = phenotype.clone()
        # Note: CV2 operations remain on CPU as cv2.cuda module would need separate implementation
        overlay_cpu = overlay.cpu().numpy()
        cv2.circle(
            overlay_cpu,
            center=(int(row[1].item() * w), int(row[0].item() * w)),
            radius=int(row[2].item() * radius_avg),
            color=(int(row[3].item() * 255), int(row[4].item() * 255), int(row[5].item() * 255)),
            thickness=-1,
        )
        overlay = torch.from_numpy(overlay_cpu).to(device)
        alpha = row[6]
        phenotype = (overlay * alpha + phenotype * (1 - alpha))
    
    return phenotype.cpu().numpy()/255.


def mutate(soln, mut, device='cuda'):
    soln_cuda = torch.from_numpy(soln).to(device) if isinstance(soln, np.ndarray) else soln.to(device)
    new_specie = soln_cuda.clone()

    genes = soln_cuda.shape[0]
    length = soln_cuda.shape[1]

    y = torch.randint(0, genes, (1,)).item()
    change = torch.randint(0, length + 1, (1,)).item()

    if torch.rand(1).item() < (1 / mut):
        change -= max(change - 1, 1)
        i, j = y, torch.randint(0, genes, (1,)).item()
        i, j, s = (i, j, -1) if i < j else (j, i, 1)
        new_specie[i: j + 1] = torch.roll(new_specie[i: j + 1], shifts=s, dims=0)
        y = j
    
    change = max(change, 1)
    selection = torch.randperm(length)[:change]

    if torch.rand(1).item() < mut:
        new_specie[y, selection] = torch.rand(len(selection), device=device)
    else:
        new_specie[y, selection] += (torch.rand(len(selection), device=device) - 0.5) / 3
        new_specie[y, selection] = torch.clamp(new_specie[y, selection], 0, 1)

    return new_specie


class Attack:
    def __init__(self, params):
        self.params = params
        self.process = []
        self.device = params["device"]  

    def save_image(self, np_img, output_dir):
        if isinstance(np_img, torch.Tensor):
            np_img = np_img.cpu().numpy()
        np_img = (np_img * 255).astype(np.uint8)
        img = Image.fromarray(np_img, mode='RGB')
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        img.save(output_dir)
    
    def convert_to_serializable(self, data):
        if isinstance(data, dict):
            return {key: self.convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_to_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()  
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        else:
            return data 

    def completion_procedure(self, adversarial, x_adv, queries, loc, patch, loss_function):
        data = {
            "orig": self.params["x1"],
            "second": self.params["x2"],
            "adversary": x_adv,
            "adversarial": adversarial,
            "queries": queries,
            "loc": loc,
            "patch": patch,
            "patch_width": int(math.ceil(self.params["eps"] ** .5)),
            "final_prediction": loss_function(x_adv, self.params["x2"])[0],
            "process": self.process
        }
        data_final = {
            "population": self.params["N"],
            "adv_img": x_adv,
            "adv_score": loss_function(x_adv, self.params["x2"])[1],
            "l2": l2(adv_patch=patch, orig_patch=self.params["x1"][loc[0]: loc[0] + patch.shape[0], loc[1]: loc[1] + patch.shape[1], :].clone()),
        }

        save_folder = self.params["save_directory"]
        save_result_pickle_path = os.path.join(save_folder, "result_pickle_")
        save_image_path = os.path.join(save_folder, "result_image_")
        save_final_path = os.path.join(save_folder, "result_final_")
        serializable_data = self.convert_to_serializable(data_final)

        with open(f"{save_result_pickle_path}{adversarial}.pkl", 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        with open(f"{save_final_path}{adversarial}.json", 'w') as final_file:
            json.dump(serializable_data, final_file, indent=4)
        self.save_image(np.array(data['orig'].cpu()), f"{save_image_path}orig.png")

        self.save_image(np.array(data['adversary']), f"{save_image_path}{adversarial}.png")

    
    def optimise(self, loss_function):
        # Initialize
        x = self.params["x1"]
        x2 = self.params["x2"]
        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        eps = self.params["eps"]
        s = int(math.ceil(eps ** .5))

        patch_geno = torch.rand(self.params["N"], 7, device=self.device)
        patch = render(patch_geno, s, device=self.device)
        loc = torch.randint(h - s, size=(2,), device=self.device)
        
        x_adv = x.clone()
        patch_tensor = torch.from_numpy(patch).to(self.device) if isinstance(patch, np.ndarray) else patch
        x_adv[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch_tensor
        x_adv = torch.clamp(x_adv, 0., 1.)

        adversarial, loss = loss_function(x_adv, x2)

        l2_curr = l2(
            adv_patch=patch_tensor,
            orig_patch=x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :],
            device=self.device
        )

        saved_it = 0
        patch_counter = 0
        update_loc_period = self.params["update_loc_period"]
        n_queries = self.params["n_queries"]

        for it in tqdm(range(1, n_queries)):
            patch_counter += 1
            # self.save_image(np.array(x_adv.cpu().numpy()), r"D:\codePJ\RESEARCH\GECCO2025\CamoPatch\results\result_image_False.png")
            if patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"])
                patch_new = render(patch_new_geno, s, device=self.device)
                patch_new_tensor = torch.from_numpy(patch_new).to(self.device) if isinstance(patch_new, np.ndarray) else patch_new

                x_adv_new = x.clone()
                x_adv_new[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch_new_tensor
                x_adv_new = torch.clamp(x_adv_new, 0., 1.)

                adversarial_new, loss_new = loss_function(x_adv_new, x2)

                orig_patch = x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].clone()
                l2_new = l2(adv_patch=patch_new_tensor, orig_patch=orig_patch, device=self.device)

                if adversarial and adversarial_new:
                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new
                else:
                    if loss_new > loss:
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new
            else:
                patch_counter = 0
                sh_i = int(max(sh_selection(n_queries, it) * h, 0))
                loc_new = loc.clone()
                loc_new = update_location(loc_new.cpu().numpy(), sh_i, h, s)
                loc_new = torch.tensor(loc_new, device=self.device)

                x_adv_new = x.clone()
                x_adv_new[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :] = patch_tensor
                x_adv_new = torch.clamp(x_adv_new, 0., 1.)

                adversarial_new, loss_new = loss_function(x_adv_new, x2)

                orig_patch_new = x[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :].clone()
                l2_new = l2(adv_patch=patch_tensor, orig_patch=orig_patch_new, device=self.device)

                if adversarial and adversarial_new:
                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new
                else:
                    diff = loss_new - loss
                    curr_temp = self.params["temp"] / (it + 1)
                    metropolis = math.exp(-diff / curr_temp)

                    if loss_new > loss or torch.rand(1).item() < metropolis:
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new

            self.process.append([loc.cpu().numpy(), patch_geno.cpu().numpy(), l2_curr, loss])
            saved_it = it

        x_adv_cpu = x_adv.cpu().numpy()
        loc_cpu = loc.cpu().numpy()
        patch_cpu = patch_tensor.cpu().numpy() if isinstance(patch, torch.Tensor) else patch

        self.completion_procedure(adversarial, x_adv_cpu, saved_it, loc_cpu, patch_cpu, loss_function)

        return