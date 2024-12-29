from population import Population
from fitness import Fitness
from individual import Individual
import torch
import random
from torchvision.utils import save_image


class GA:
    def __init__(self, n_iter: int, 
                 population: 'Population', 
                 fitness: 'Fitness',
                 tourament_size: int, ):
        self.n_iter = n_iter
        self.pop = population
        self.args = self.pop.get_params
        self.tourament_size = tourament_size
        self.fitness = fitness
        
    def solve(self):
        P = self.pop.P
        for i in range(self.n_iter):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)                
                offstring_1, offstring_2 = parent1.crossover(parent2)
                
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.tourament_selection(O_P)
            self.pop.P = P
            self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
        return self.pop            
    
    
    def has_converged(self, population: list['Individual']) -> bool:
        """
        Check if all patches in the population are identical.
        If they are, the population has converged.
        """
        first_patch = population[0].patch
        for individual in population[1:]:
            if not torch.equal(first_patch, individual.patch):  # Check if patches are identical
                return False
        return True
    
         
    def tourament_selection(self, pool: list['Individual']) -> list['Individual']:
        pool_fitness, _, _= self.fitness.benchmark(pool)
        winner = []
        for i in range(0, len(pool), self.tourament_size):
            idx = i + torch.argmax(pool_fitness[i:i+self.tourament_size])
            winner.append(pool[idx])
            
        return winner
    
    def save_best(self, P: list['Individual']) -> None:
        fitness, adv_scores, psnr_scores = self.fitness.benchmark(P)
        best_idx = torch.argmax(fitness)
        best_patch = P[best_idx]
        best_adv_img = self.fitness.apply_patch_to_image(best_patch.patch, best_patch.location)
        
        print("Best_adv: ", adv_scores[best_idx])
        save_image(best_adv_img, 'process.png')
                
                

        