import numpy as np

from model.population import Population
from operators.crossover import Crossover
from utils import check_valid, get_hashKey

def crossover(parent_1, parent_2, typeC, **kwargs):
    
    # parent_1 , parent_2 -> 2 offstring
    
    # problem_name = kwargs['problem_name']
    offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()

    if typeC == '1X':  # 1-point crossover
        point = np.random.randint(1, len(parent_1))

        offspring_1[point:], offspring_2[point:] = offspring_2[point:], offspring_1[point:].copy()

    elif typeC == '2X':  # 2-point crossover
        # if problem_name == 'NASBench301':
        #     points_list = np.random.choice(range(2, len(parent_1) - 1, 2), 2, replace=False)
        # else:
        points_list = np.random.choice(range(1, len(parent_1) - 1), 2, replace=False)

        lower_pt = min(points_list)
        upper_pt = max(points_list)

        offspring_1[lower_pt: upper_pt], offspring_2[lower_pt: upper_pt] = \
            offspring_2[lower_pt: upper_pt], offspring_1[lower_pt: upper_pt].copy()

    elif typeC == 'UX':  # Uniform crossover
        pts = np.random.randint(0, 2, parent_1.shape, dtype=bool)

        offspring_1[pts], offspring_2[pts] = offspring_2[pts], offspring_1[pts].copy()
    
    elif typeC == 'SE': 
            segment_start = np.random.randint(0, len(parent_1) - 1)
            segment_length = np.random.randint(1, len(parent_1) - segment_start)
            segment_end = segment_start + segment_length

            offspring_1[segment_start:segment_end], offspring_2[segment_start:segment_end] = \
                offspring_2[segment_start:segment_end], offspring_1[segment_start:segment_end].copy()

    
    return [offspring_1, offspring_2]


class PointCrossover(Crossover):
    def __init__(self, method=None):
        super().__init__(n_parents=2, prob=0.9)
        available_methods = ['1X', '2X', 'UX', 'SE']
        if method not in available_methods:
            raise ValueError('Invalid crossover method: ' + method)
        else:
            self.method = method

    def _do(self, problem, P, **kwargs):
        problem_name = 'cc'
        offspring_size = len(P)
        O = Population(offspring_size) # emty pop with size

        n = 0
        nCrossovers, maxCrossovers = 0, offspring_size * 5

        while True:
            I = np.random.choice(offspring_size, size=(offspring_size // 2, self.n_parents), replace=False)
            P_ = P[I]
            for i in range(len(P_)):
                if np.random.random() < self.prob:
                    o_X = crossover(P_[i][0].X, P_[i][1].X, self.method, problem_name=problem_name)
                    for j, X in enumerate(o_X):
                        if maxCrossovers - nCrossovers > 0:
                            O[n].set('X', o_X[j])
                            n += 1
                            if n - offspring_size == 0:
                                return O
                
                else:
                    for o in P_[i]:
                        O[n].set('X', o.X)
                        n += 1
                        if n - offspring_size == 0:
                            return O
            nCrossovers += 1


if __name__ == '__main__':
    pass
