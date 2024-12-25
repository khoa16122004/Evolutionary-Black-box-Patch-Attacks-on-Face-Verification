from problems import Problem
from .fitness import Fitness
import numpy as np


class PatchFaceAttack(Problem):
    
    def __init__(self, max_eval, location, MODEL, img1_np, img2_np, label, patch_size):
        # super.__init__(max_eval, "PatchAttack")
        self.max_eval = max_eval
        self.fit = Fitness(location, MODEL, img1_np, img2_np, label, patch_size)
        self.available_ops = list(np.arange(0, 256))
    def _evaluate(self, designs, *args, **kwargs):
        adv_scores, fsnr_scores = self.fit.benchmark(designs)
        
        return adv_scores, fsnr_scores, 0.0, 0.0