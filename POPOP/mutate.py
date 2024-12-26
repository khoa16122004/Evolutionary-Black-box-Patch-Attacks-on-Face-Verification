import numpy as np
import random
import cv2 as cv

class Mutation:
    def __init__(self, patch_size, **kwargs):
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
        return image_with_shapes
    
    def __call__(self, patch, number_of_shapes=2):
        patch_final = self.add_random_shape_to_image(patch, number_of_shapes)
        return patch_final.flatten()

if __name__ == "__main__":
    number_of_individuals = 1
    patch_size = 3
    mutation = Mutation(patch_size=patch_size)
    patch = np.random.randint(0, 10, (3 * patch_size * patch_size))
    print("patch: ", patch.shape)
    patch_mutated = mutation(patch)
    print("patch mutated: ", patch_mutated.shape)
    print("patch mutated: ", patch_mutated.flatten().shape)