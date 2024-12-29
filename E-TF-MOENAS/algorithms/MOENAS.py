"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
import time

from algorithms import Algorithm
from model.individual import Individual
import numpy as np

INF = 9999999
class NSGAII(Algorithm):
    """
    NSGA-Net
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f0, self.f1 = None, None
        self.individual = Individual(rank=INF, crowding=-1)

    def _setup(self):
        pass

    def _evaluate(self, pop):
        """
        Call function *problem.evaluate* to evaluate the fitness values of solutions.
        """
        self.finish_executed_time_algorithm = time.time()
        self.executed_time_algorithm_history.append(
            self.executed_time_algorithm_history[-1] + (self.finish_executed_time_algorithm - self.start_executed_time_algorithm))

        # comp_metric, perf_metric, benchmark_time, indicator_time = self.problem.evaluate(arch=arch, comp_metric=self.f0,
        #                                                                                  perf_metric=self.perf_metric,
        #                                                                                  epoch=self.epoch,
        #                                                                                  subset=self.subset)
        
        # print(pop.shape)
        adv_scores, fsnr_scores, benchmark_time, indicator_time = self.problem.evaluate(pop.get('X'))
        
        scores = np.stack((adv_scores, fsnr_scores), axis=1)
                
        self.n_eval += len(adv_scores)
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + benchmark_time)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + benchmark_time + indicator_time)
        self.running_time_history.append(self.evaluated_time_history[-1] + self.tmp + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        
        # print(scores)
        return scores

    def _initialize(self):
        """
        Workflow in 'Initialization' step:
        + Sampling 'pop_size' architectures.
        + For each architecture, evaluate its fitness.
            _ Update the Elitist Archive (search).
        """
        P = self.sampling.do(self.problem)
        F = self.evaluate(P)
        P.set('F', F)
        for i in range(self.pop_size):
            self.E_Archive_search.update(P[i], algorithm=self)
        self.pop = P
        print('Initialized - Done')

    def _mating(self, P):
        """
         Workflow in 'Mating' step:
        + Create the offspring throughout 'crossover' step.
        + Perform 'mutation' step on the offspring.
        """
        O = self.crossover.do(self.problem, P, algorithm=self)

        O = self.mutation.do(self.problem, P, O, algorithm=self)
        O_F = self.evaluate(O)
        O.set('F', O_F)
        for i in range(len(O)):
            self.E_Archive_search.update(O[i], algorithm=self)

        return O

    def _next(self, pop):
        """
         Workflow in 'Next' step:
        + Create the offspring.
        + Select the new population.
        """
        offsprings = self._mating(pop)
        pool = pop.merge(offsprings)
        pop = self.survival.do(pool, self.pop_size)
        self.pop = pop

    def log_elitist_archive(self, **kwargs):
        self.nEvals_history.append(self.n_eval)

        elitist_archive_search = {
            'X': self.E_Archive_search.X.copy(),
            'F': self.E_Archive_search.F.copy(),
        }
        self.E_Archive_search_history.append(elitist_archive_search)

if __name__ == '__main__':
    pass
