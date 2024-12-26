import numpy as np

class Crossover:
    def __init__(self, type='SE'):
        self.type = type
    def __call__(self, parent1, parent2):
        offspring_1 = parent1.copy()
        offspring_2 = parent2.copy()
        if self.type == 'SE':
            segment_start = np.random.randint(0, len(parent1) - 1)
            segment_length = np.random.randint(1, len(parent1) - segment_start)
            segment_end = segment_start + segment_length

            offspring_1[segment_start:segment_end], offspring_2[segment_start:segment_end] = \
                offspring_2[segment_start:segment_end], offspring_1[segment_start:segment_end].copy()
        elif self.type == 'CM': # cross match
            crossover_point1 = np.random.randint(0, len(parent1))
            crossover_point2 = np.random.randint(0, len(parent2))
            offspring_1 = np.concatenate([parent1[:crossover_point1], parent2[crossover_point1:]])
            offspring_2 = np.concatenate([parent2[:crossover_point2], parent1[crossover_point2:]])
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