import numpy as np
import random

class Crossover:
    def __init__(self, type='SE', patch_size=20):
        self.type = type
        self.patch_size = patch_size
        
    def convert_to_3d(self, flattened_patches):
        if flattened_patches.shape[0] == 1:
            return flattened_patches[0].reshape(self.patch_size, self.patch_size, 3)
        return np.array([patch.reshape(self.patch_size, self.patch_size, 3) for patch in flattened_patches])
    
    def __call__(self, parent1, parent2):

        
        types = ['SE', 'CM', 'y_swap', 'x_swap', 'Random_SE']
        random_type = random.choice(types)
        
        random_type = 'Random_SE'
        
        if random_type == 'SE':
            offspring_1 = parent1.copy()
            offspring_2 = parent2.copy()
            segment_start = np.random.randint(0, len(parent1) - 1)
            segment_length = np.random.randint(1, len(parent1) - segment_start)
            segment_end = segment_start + segment_length

            offspring_1[segment_start:segment_end], offspring_2[segment_start:segment_end] = \
                offspring_2[segment_start:segment_end], offspring_1[segment_start:segment_end].copy()
       
        elif random_type == 'CM': # cross match
            offspring_1 = parent1.copy()
            offspring_2 = parent2.copy()
            
            crossover_point1 = np.random.randint(0, len(parent1))
            crossover_point2 = np.random.randint(0, len(parent2))
            offspring_1 = np.concatenate([parent1[:crossover_point1], parent2[crossover_point1:]])
            offspring_2 = np.concatenate([parent2[:crossover_point2], parent1[crossover_point2:]])

        elif random_type == "x_swap":
            parent1_3d = self.convert_to_3d(np.array([parent1]))
            parent2_3d = self.convert_to_3d(np.array([parent2]))
            
            offspring_1 = parent1_3d.copy()
            offspring_2 = parent2_3d.copy()
            cut_point = random.randint(0, self.patch_size)
            
            offspring_1[:cut_point, :, :] = parent2_3d[:cut_point, :, :]
            offspring_2[:cut_point, :, :] = parent1_3d[:cut_point, :, :]
        
            offspring_1 = offspring_1.flatten()
            offspring_2 = offspring_2.flatten()
        
        elif random_type == "y_swap":
            parent1_3d = self.convert_to_3d(np.array([parent1]))
            parent2_3d = self.convert_to_3d(np.array([parent2]))
            
            offspring_1 = parent1_3d.copy()
            offspring_2 = parent2_3d.copy()
            
            cut_point = random.randint(0, self.patch_size)
                    
            offspring_1[:, :cut_point, :] = parent2_3d[:, :cut_point, :]
            offspring_2[:, :cut_point, :] = parent1_3d[:, :cut_point, :]
            
            offspring_1 = offspring_1.flatten()
            offspring_2 = offspring_2.flatten()
        
        
        elif random_type == "Random_SE":
            offspring_1 = parent1.copy()
            offspring_2 = parent2.copy()
            
            n_segments = np.random.randint(2, 6)
            
            cut_points = sorted(np.random.choice(range(1, len(parent1)), n_segments-1, replace=False))
            cut_points = [0] + list(cut_points) + [len(parent1)]
            
            for i in range(len(cut_points) - 1):
                if np.random.random() < 0.5:  # 50% cơ hội swap
                    start = cut_points[i]
                    end = cut_points[i+1]
                    offspring_1[start:end], offspring_2[start:end] = \
                        offspring_2[start:end].copy(), offspring_1[start:end].copy()
        
        return [offspring_1, offspring_2]
    
if __name__ == "__main__":
    SE = Crossover('SE')
    CM = Crossover('CM')
    number_of_individuals = 1
    patch_size = 3
    
    parent1 = np.random.randint(0, 10, (3 * patch_size * patch_size))
    parent2 = np.random.randint(0, 10, (3 * patch_size * patch_size))
    print(f"parent1: {parent1}\nparent2: {parent2}")
    # TEST SE
    print("TEST SE")
    print(SE(parent1, parent2))

    # TEST CM
    print("TEST CM")
    print(CM(parent1, parent2))