import numpy as np
from typing import Callable, List, Tuple
import random
from tqdm import tqdm
from .visualization import save_image_with_patch
class POPOP:
    def __init__(self, 
                 patch_size: int,
                population_size: np.ndarray,
                fitness_func: Callable[[np.ndarray], float],
                crossover_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                mutation_func: Callable[[np.ndarray], np.ndarray],
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.8,
                tournament_size: int = 4,
                visual: bool = False):
        """
        Khởi tạo POP-OP
        
        Parameters:
        - fitness_func: Hàm đánh giá độ thích nghi
        - crossover_func: Hàm lai ghép
        - mutation_func: Hàm đột biến
        - population_size: Kích thước quần thể
        - mutation_rate: Tỷ lệ đột biến
        - crossover_rate: Tỷ lệ lai ghép
        - tournament_size: Kích thước tournament selection
        """
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size  
        self.visual = visual
        self.population = self.initialize_population(patch_size)
        self.fitness_values = np.zeros(self.population_size)
        self.patch_size = patch_size
    
    def initialize_population(self, patch_size) -> np.ndarray:
        """Khởi tạo quần thể các patch 1D"""
        return np.random.randint(0, 256, (self.population_size, patch_size**2*3))

    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Đánh giá độ thích nghi của toàn bộ quần thể input"""
        # fitness_values = np.zeros(len(population))
        # for i in range(len(population)):
        #     fitness_values[i] = self.fitness_func(population[i])
        fitness_values = self.fitness_func(population)
        return fitness_values
    
    def _tournament_selection(self, pool, pool_fitness) -> np.ndarray:
        winner = []
        for i in range(0, len(pool), self.tournament_size):
            idx = i + np.argmax(pool_fitness[i:i+self.tournament_size])
            winner.append(pool[idx])
        return winner
    
    def evolve(self, n_generations: int) -> Tuple[np.ndarray, float]:
        """        
        Returns:
        - best_solution: Cá thể tốt nhất
        - best_fitness: Giá trị fitness tốt nhất
        """
        all_gen = 0
        for generation in tqdm(range(n_generations)):
            all_gen += 1
            self.fitness_values = self._evaluate_fitness(self.population)
            if np.all(np.all(self.population == self.population[0], axis=1)):
                break
            if np.all(self.fitness_values == self.fitness_values[0]):
                break
            # Tạo quần thể mới
            offspring  = []

            # Tạo quàn thể con 0
            while len(offspring) < self.population_size:
                # Chọn cha mẹ random từ quẩn thể P
                parent1 = random.choice(self.population)
                while True:
                    parent2 = random.choice(self.population)
                    if not np.array_equal(parent2, parent1):
                        break

                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover_func(parent1, parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                if random.random() < self.mutation_rate:
                    child1 = self.mutation_func(child1)
                    child2 = self.mutation_func(child2)
                
                offspring.append(child1)
                offspring.append(child2)

            pool = self.population.copy()
            pool = np.concatenate((pool, offspring), axis=0)

            new_population = []
            for _ in range(2):
                np.random.shuffle(pool)
                new_selection = self._tournament_selection(pool, self._evaluate_fitness(pool))
                for new in new_selection:
                    new_population.append(new)
                            
            new_population = np.array(new_population)
            self.population = new_population
            if self.visual:
                best_idx = np.argmax(self.fitness_values)
                best_patch = self.population[best_idx]
                best_patch = best_patch.reshape(self.patch_size, self.patch_size, 3)
                save_image_with_patch(self.fitness_func.get_img1(), self.fitness_func.get_location(), best_patch, f"patch_attack_result")
        print(f"all_gen: {all_gen}")
        self.fitness_values = self._evaluate_fitness(self.population)
        best_idx = np.argmax(self.fitness_values)
        
        return self.population[best_idx], self.fitness_values[best_idx]

if __name__ == "__main__":
    def mock_fitness_func(x):
        return -np.sum(x**2)
    def mock_crossover_func(parent1, parent2):
        crossover_point = np.random.randint(0, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    def mock_mutation_func(child):
        mutation_point = np.random.randint(0, len(child))
        child[mutation_point] = np.random.randint(0, 256)
        return child
    def mock_population(patch_size=2, number_of_individuals=12):
        patches = np.random.randint(0, 10, (number_of_individuals, patch_size, patch_size, 3))
        flattened_patches = patches.reshape(number_of_individuals, -1)
        return flattened_patches
    
    mock = mock_population(patch_size=2, number_of_individuals=6)
    # print(mock.shape)

    mock_popop = POPOP(fitness_func=mock_fitness_func,
                  tournament_size=4,
                    population=mock,
                    crossover_func=mock_crossover_func,
                    mutation_func=mock_mutation_func,
                    )
    best_solution, best_fitness = mock_popop.evolve(300)
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"fitness: {mock_popop.fitness_values}")
    