import numpy as np
from operators.mutation import Mutation
from model.population import Population
from utils import check_valid, get_hashKey


class BitStringMutation(Mutation):
    def __init__(self, **kwargs):
        super().__init__(prob=1)

    def mutation(self, problem, P, O, **kwargs):
        # problem_name = problem.name

        O_old_X = O.get('X')

        offspring_size = len(O)
        len_X = len(O_old_X[-1])

        nMutations, maxMutations = 0, offspring_size * 5

        self.prob = 1 / len_X

        O_new = Population(offspring_size)

        n = 0
        while True:
            for X_old in O_old_X:
                o_X = X_old.copy()

                for m, prob in enumerate(np.random.rand(len_X)):
                    if prob <= self.prob:
                        available_ops = problem.available_ops.copy()
                        available_ops.remove(o_X[m])
                        new_op = np.random.choice(available_ops)
                        o_X[m] = new_op

                if maxMutations - nMutations > 0:
                    O_new[n].set('X', o_X)

                    n += 1
                    if n - offspring_size == 0:
                        return O_new
            nMutations += 1

    def _do(self, problem, P, O, **kwargs):
        return self.mutation(problem, P, O, **kwargs)


if __name__ == '__main__':
    pass
