import torch
from population import Population
from individual import Individual
import torch.nn.functional as F
from torch import nn

class Fitness:
    
    def __init__(self, patch_size: int, img1:torch.Tensor, img2: torch.Tensor, model: nn.Module, label: int, recons_w: float, attack_w: float, fitness_type: str) -> None:
        self.img1 = img1.cuda()
        self.img2_feature = model(img2.cuda().unsqueeze(0))
        self.model = model.eval()
        self.patch_size = patch_size
        
        self.attack_w = attack_w
        self.recons_w = recons_w
        self.label = label
        self.fitness_type = fitness_type
        self.max_psnr, self.min_psnr = None, None
        self.max_adv, self.max_adv = None, None

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
           
           # if self.fitness_type == "adaptive":
            #    adv_scores = torch.where(adv_scores > 0, torch.tensor(0.0, device=adv_scores.device), adv_scores)
            
            return adv_scores
            
    def evaluate_psnr(self, P: list['Individual']) -> torch.Tensor:
        adv_imgs = torch.stack([self.apply_patch_to_image(ind.patch, ind.location) for ind in P])
        mse = F.mse_loss(adv_imgs, self.img1.expand_as(adv_imgs), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1) 
        psnr_scores = torch.log10(1 / (mse + 1e-8))
        
        return psnr_scores / 10
    
    def update_min_max(self, adv_scores: torch.Tensor, psnr_scores:torch.Tensor) -> None: 
        self.min_psnr = torch.min(psnr_scores.min(), self.min_psnr)
        self.max_psnr = torch.max(psnr_scores.max(), self.max_psnr)
        self.min_adv = torch.min(adv_scores.min(), self.min_adv)
        self.max_adv = torch.max(psnr_scores.min(), self.min_adv)
 

    def benchmark(self, P: list['Individual']) -> torch.Tensor:
        adv_scores = self.evaluate_adv(P)
        psnr_scores = self.evaluate_psnr(P)
        
        for i in range(len(P)):
            P[i].adv_score = adv_scores[i]
            P[i].psnr_score = psnr_scores[i]
        #if self.fitness_type == "normalize":
            # normalize each score
            # self.update_min_max(adv_scores, psnr_scores)
            # adv_scores_normalize = (adv_scores - self.min_adv) / (self.max_adv - self.min_adv)
            # psnr_scores_normalize = (adv_scores - self.min_psnr) / (self.max_psnr - self.min_psnr)
            # return adv_scores_normalize + psnr_scores_normalize, adv_scores, psnr_scores
        if self.fitness_type == "adaptive":
                adv_scores = torch.where(adv_scores > 0, torch.tensor(0.0, device=adv_scores.device), adv_scores)

        return adv_scores * self.attack_w + psnr_scores * self.recons_w, adv_scores, psnr_scores
        
        
