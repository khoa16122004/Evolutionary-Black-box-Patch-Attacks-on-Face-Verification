from model.population import Population
# from utils import get_hashKey
import random
import numpy as np
import cv2 as cv

class RandomSampling:

    def __init__(self, n_sample=0, patch_size=20):
        self.n_sample = n_sample
        self.precomputed_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(1000)]
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
                0.8 * random.randint(0, 255),
                0.8 * random.randint(0, 255),
                0.8 * random.randint(0, 255)
            )
            thickness = 1

            # if shape_type == 'circle':
            #     center = (
            #         random.randint(0, width - 1),
            #         random.randint(0, height - 1)
            #     )
            #     radius = random.randint(5, min(width, height) // 4)
            #     cv.circle(image_with_shapes, center, radius, color, thickness)

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

            # elif shape_type == 'ellipse':
            #     center = (
            #         random.randint(0, width - 1),
            #         random.randint(0, height - 1)
            #     )
            #     axes = (
            #         random.randint(5, width // 4),
            #         random.randint(5, height // 4)
            #     )
            #     angle = random.randint(0, 360)
            #     start_angle = 0
            #     end_angle = 360
            #     cv.ellipse(image_with_shapes, center, axes, angle, start_angle, end_angle, color, thickness)

            # elif shape_type == 'line':
            #     pt1 = (
            #         random.randint(0, width - 1),
            #         random.randint(0, height - 1)
            #     )
            #     pt2 = (
            #         random.randint(0, width - 1),
            #         random.randint(0, height - 1)
            #     )
            #     cv.line(image_with_shapes, pt1, pt2, color, thickness)

        return image_with_shapes
    
        
    def get_next_color(self):
        color = self.precomputed_colors[0]
        return color
    
    def create_random_population(self):
        patchs = [np.zeros(self.patch_size**2 * 3, dtype=np.uint8) for _ in range(self.n_sample)]
        return np.array([self.add_random_shape_to_image(patch, 2).flatten() for patch in patchs])
        
        
    def do(self, problem, **kwargs):        
        P = Population(self.n_sample)
        X = self.create_random_population()
        P.set('X', X)
        return P
        
        
        

if __name__ == '__main__':
    a = RandomSampling(n_sample=5)
    a.do()
    pass