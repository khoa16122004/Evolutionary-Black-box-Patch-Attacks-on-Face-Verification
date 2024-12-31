import torch
from population import Population
from individual import Individual
import torch.nn.functional as F
from torch import nn

class Fitness:
    
    def __init__(self, patch_size: int, img1:torch.Tensor, img2: torch.Tensor, model: nn.Module, label: int, recons_w: float, attack_w: float) -> None:
        self.img1 = img1.cuda()
        self.img2_feature = model(img2.cuda().unsqueeze(0))
        self.model = model.eval()
        self.patch_size = patch_size
        
        self.attack_w = attack_w
        self.recons_w = recons_w
        self.label = label
        
    def apply_patch_to_image(self, patch: torch.Tensor, location: tuple[int, int, int, int]):
        img_copy = self.img1.clone()
        x_min, x_max, y_min, y_max = location
        img_copy[:, x_min : x_max, y_min : y_max] = patch
        return img_copy
    
        
    def evaluate_adv(self, P: list['Individual']):
        adv_imgs = torch.stack([self.apply_patch_to_image(ind.patch, ind.location) for ind in P])
        
        with torch.no_grad():
            adv_batch = adv_imgs.cuda()
            adv_features = self.model(adv_batch)
            sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
            adv_scores = (1 - self.label) * (0.5 - sims) + self.label * (sims - 0.5)
            
            adv_scores = torch.where(adv_scores > 0, torch.tensor(0.0, device=adv_scores.device), adv_scores)
            
            return adv_scores
            
    def evaluate_psnr(self, P: list['Individual']) -> torch.Tensor:
        adv_imgs = torch.stack([self.apply_patch_to_image(ind.patch, ind.location) for ind in P])
        mse = F.mse_loss(adv_imgs, self.img1.expand_as(adv_imgs), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1) 
        psnr_scores = torch.log10(1 / (mse + 1e-8))
        
        return psnr_scores / 10
    
    def benchmark(self, P: list['Individual']) -> torch.Tensor:
        adv_scores = self.evaluate_adv(P)
        psnr_scores = self.evaluate_psnr(P)

        return adv_scores * self.attack_w + psnr_scores * self.recons_w, adv_scores, psnr_scores
        
        