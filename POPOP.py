import numpy as np
from typing import Callable, List, Tuple
import random

class POPOP:
    def __init__(self, 
                population: np.ndarray,
                fitness_func: Callable[[np.ndarray], float],
                crossover_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                mutation_func: Callable[[np.ndarray], np.ndarray],
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.8,
                tournament_size: int = 4):
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
        self.population_size = len(population)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size  
        
        self.population = population
        self.fitness_values = np.zeros(self.population_size)
            
    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Đánh giá độ thích nghi của toàn bộ quần thể input"""
        fitness_values = np.zeros(len(population))
        for i in range(len(population)):
            fitness_values[i] = self.fitness_func(population[i])
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
        for generation in range(n_generations):
            all_gen += 1
            if np.all(np.all(self.population == self.population[0], axis=1)):
                break

            # Tạo quần thể mới
            offprings = []

            # Tạo quàn thể con 0
            while len(offprings) < self.population_size:
                # Chọn cha mẹ random từ quẩn thể P
                parent1 = random.choice(self.population)
                while True:
                    parent2 = random.choice(self.population)
                    if not np.array_equal(parent2, parent1):
                        break

                # Lai ghép va đột biến
                if random.random() < self.crossover_rate:
                    child = self.crossover_func(parent1, parent2)
                else:
                    child = parent1.copy()
                if random.random() < self.mutation_rate:
                    child = self.mutation_func(child)
                
                offprings.append(child)

            pool = self.population.copy()
            pool = np.concatenate((pool, offprings), axis=0)

            new_population = []
            for _ in range(2):
                np.random.shuffle(pool)
                new_selection = self._tournament_selection(pool, self._evaluate_fitness(pool))
                for new in new_selection:
                    new_population.append(new)
                            
            new_population = np.array(new_population)
            self.population = new_population
        print(f"all_gen: {all_gen}")
        self.fitness_values = self._evaluate_fitness(self.population)
        best_idx = np.argmax(self.fitness_values)
        
        return self.population[best_idx], self.fitness_values[best_idx]

def __main__():
    def fitness_func(x):
        return -np.sum(x**2)

    def crossover_func(parent1, parent2):
        crossover_point = np.random.randint(0, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def mutation_func(child):
        mutation_point = np.random.randint(0, len(child))
        child[mutation_point] = np.random.randint(0, 256)
        return child
    def mock_population(patch_size=2, number_of_individuals=12):
        patches = np.random.randint(0, 10, (number_of_individuals, patch_size, patch_size, 3))
        flattened_patches = patches.reshape(number_of_individuals, -1)
        return flattened_patches
    
    mock = mock_population(patch_size=2, number_of_individuals=6)
    print(mock.shape)

    popop = POPOP(fitness_func=fitness_func,
                  tournament_size=4,
                    population=mock,
                    crossover_func=crossover_func,
                    mutation_func=mutation_func,
                    )
    best_solution, best_fitness = popop.evolve(300)
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"fitness: {popop.fitness_values}")
    
__main__()