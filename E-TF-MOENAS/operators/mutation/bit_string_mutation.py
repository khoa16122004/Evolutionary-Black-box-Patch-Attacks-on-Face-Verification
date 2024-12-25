import numpy as np
from operators.mutation import Mutation
from model.population import Population
from utils import check_valid, get_hashKey

import random
import cv2 as cv

class BitStringMutation(Mutation):
    def __init__(self, patch_size, **kwargs):
        super().__init__(prob=1)
        self.patch_size = patch_size

    def convert_to_3d(self, flattened_patches):
        if flattened_patches.shape[0] == 1:
            return flattened_patches[0].reshape(self.patch_size, self.patch_size, 3)
        return np.array([patch.reshape(self.patch_size, self.patch_size, 3) for patch in flattened_patches])
   
    def add_random_shape_to_image(self, patch, number_of_shapes):
        patch_3d = self.convert_to_3d(np.array([patch]))
        image_with_shapes = patch_3d.copy()
        height, width = patch_3d.shape[:2]

        shape_types =  ['rectangle', 'line']

        for _ in range(number_of_shapes):
            shape_type = random.choice(shape_types)

            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            thickness = random.randint(1, 5)

            if shape_type == 'rectangle':
                pt1 = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                pt2 = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                cv.rectangle(image_with_shapes, pt1, pt2, color, thickness)


            elif shape_type == 'line':
                pt1 = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                pt2 = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                cv.line(image_with_shapes, pt1, pt2, color, thickness)

        return image_with_shapes # 3d

    def mutation(self, problem, P, O, **kwargs):
        O_old_X = O.get('X')

        offspring_size = len(O)
        len_X = len(O_old_X[-1])

        nMutations, maxMutations = 0, offspring_size * 5

        self.prob = 0.2

        O_new = Population(offspring_size)

        n = 0
        while True:
            for X_old in O_old_X:
                o_X = X_old.copy()

                for m, prob in enumerate(np.random.rand(len_X)):
                    if prob <= self.prob:
                        # available_ops = problem.available_ops.copy()
                        # available_ops.remove(o_X[m])
                        # new_op = np.random.choice(available_ops)
                        # o_X[m] = new_op
                        o_X = self.add_random_shape_to_image(o_X, 2).flatten()

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
