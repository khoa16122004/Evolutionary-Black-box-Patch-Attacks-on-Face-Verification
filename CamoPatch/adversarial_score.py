import numpy as np
import torch
import math
import os
import sys
import torch.nn.functional as F

def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)

def to_pytorch(tensor_image, device):
    if isinstance(tensor_image, np.ndarray):
        tensor_image = torch.from_numpy(tensor_image)
    return tensor_image.permute(2, 0, 1).to(device)

class FaceVerification:
    def __init__(self, 
                 model, 
                 true,
                 device='cuda',
                 unormalize=False):
        self.model = model.eval().to(device)  
        self.true = true
        self.device = device
        self.unormalize = unormalize
    
    def cosine_similarity(self, vec1, vec2):
        # Chuyển numpy arrays sang tensors trên device
        vec1_tensor = torch.from_numpy(vec1).to(self.device)
        vec2_tensor = torch.from_numpy(vec2).to(self.device)
        return F.cosine_similarity(vec1_tensor, vec2_tensor, dim=0).cpu().numpy()

    def get_predict(self, pred, threshold=0.5):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().item()
        if (pred >= threshold):
            return 0
        return 1

    def calculate_similarity(self, preds_1, preds_2):
        similarity_scores = F.cosine_similarity(preds_1, preds_2, dim=1)
        return similarity_scores

    def get_pred(self, img1, img2):
        if self.unormalize:
            img1_ = img1 * 255.
        else:
            img1_ = img1

        img1_ = to_pytorch(img1_, self.device)
        img1_ = img1_[None, :]
        img2_ = to_pytorch(img2, self.device)
        img2_ = img2_[None, :]
        
        preds1 = self.model(img1_)
        preds2 = self.model(img2_)
        sims = self.calculate_similarity(preds1, preds2)
        y = self.get_predict(sims)
    
        return y, sims, 1-sims

    def __call__(self, img1, img2):
        y, sims, not_sims = self.get_pred(img1, img2)
        is_adversarial = True if y != self.true else False
        
        if isinstance(sims, torch.Tensor):
            adv_scores = (1 - self.true) * (0.5 - sims) + self.true * (sims - 0.5)
            adv_scores = float(adv_scores.cpu().item())
        else:
            adv_scores = (1 - self.true) * (0.5 - sims) + self.true * (sims - 0.5)

        return [is_adversarial, adv_scores]