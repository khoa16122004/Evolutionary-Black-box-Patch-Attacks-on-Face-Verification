from population import Population
from fitness import Fitness
from individual import Individual
import torch
import random
from torchvision.utils import save_image
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort

class GA:
    def __init__(self, n_iter: int, 
                 population: 'Population', 
                 fitness: 'Fitness',
                 tourament_size: int, 
                 interval_update: int,
                 crossover_type: str):
        self.n_iter = n_iter
        self.pop = population
        self.args = self.pop.get_params
        self.tourament_size = tourament_size
        self.fitness = fitness
        self.interval_update = interval_update
        self.crossover_type = crossover_type
        
    def solve(self):
        """
        Parallel update the location and content of individuals
        """
        
        P = self.pop.P
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)  
                if self.cross_type == 'blended':              
                    offstring_1, offstring_2 = parent1.crossover_blended(parent2)
                elif self.cross_type == 'UX':
                    offstring_1, offstring_2 = parent1.crossover_UX(parent2)
                
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.tourament_selection(O_P)
            self.pop.P = P
            # self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score     

    def solve_sequential(self):
        """
        Sequential update the location and content of individuals
        """
        self.prob_mutate_location = 0.0 # mutate for random_location 
        P = self.pop.P
        
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)                
                offstring_1, offstring_2 = parent1.crossover(parent2)
                
                if i % self.interval_update == 0:
                    print("Mutate location")
                    offstring_1.mutate_location()
                    offstring_2.mutate_location()
                else:
                    offstring_1.mutate_content()
                    offstring_2.mutate_content()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.tourament_selection(O_P)
            self.pop.P = P
            # self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score   
   
    
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
        
        print(f"Best_adv: {adv_scores[best_idx]}, Best_psnr: {psnr_scores[best_idx]}")
        # save_image(best_adv_img, 'process.png')
        return best_adv_img ,adv_scores[best_idx].item(), psnr_scores[best_idx].item()
        

selector = NonDominatedSorting()

class NSGAII:
    def __init__(self, n_iter: int, 
                 population: 'Population', 
                 fitness: 'Fitness',
                 tourament_size: int, 
                 interval_update: int,
                 crossover_type: str):
        self.n_iter = n_iter
        self.pop = population
        self.args = self.pop.get_params
        self.tourament_size = tourament_size
        self.fitness = fitness
        self.interval_update = interval_update
        self.crossover_type = crossover_type
        
    def solve(self):
        """
        Parallel update the location and content of individuals
        """
        
        P = self.pop.P
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)  
                if self.crossover_type == 'Blended':              
                    offstring_1, offstring_2 = parent1.crossover_blended(parent2)
                elif self.crossover_type == 'UX':
                    offstring_1, offstring_2 = parent1.crossover_UX(parent2)
                
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.selection(O_P)
            self.pop.P = P
            # self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score  


    def solve_sequential(self):
        """
        Sequential update the location and content of individuals
        """
        self.prob_mutate_location = 0.0 # mutate for random_location 
        P = self.pop.P
        
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)                
                offstring_1, offstring_2 = parent1.crossover(parent2)
                
                if i % self.interval_update == 0:
                    print("Mutate location")
                    offstring_1.mutate_location()
                    offstring_2.mutate_location()
                else:
                    offstring_1.mutate_content()
                    offstring_2.mutate_content()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.selection(O_P)
            
            self.pop.P = P
            # self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score  
   
    
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
    
    
    def selection(self, pool: list['Individual']) -> list['Individual']:
        _, adv_scores, fsnr_scores = self.fitness.benchmark(pool)
        F = np.array(torch.stack([adv_scores, fsnr_scores], dim=1).cpu().detach())
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=self.pop.pop_size)
        survivors = []

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calculating_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pool[i].rank = k
                pool[i].crowding = crowding_of_front[j]

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > self.pop.pop_size:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(self.pop.pop_size - len(survivors))]
 
            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
            
        return [pool[i] for i in survivors]

    
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
        
        print(f"Best_adv: {adv_scores[best_idx]}, Best_psnr: {psnr_scores[best_idx]}")
        # save_image(best_adv_img, 'process.png')
        return best_adv_img ,adv_scores[best_idx].item(), psnr_scores[best_idx].item()
        
                
def calculating_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    return crowding



                