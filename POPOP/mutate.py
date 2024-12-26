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

        shape_types = ['rectangle', 'line', 'circle', 'gaussian_noise', 'color_blending']

        for _ in range(number_of_shapes):
            shape_type = random.choice(shape_types)

            color = (
                0.8 * random.randint(0, 255),
                0.8 * random.randint(0, 255),
                0.8 * random.randint(0, 255)
            )
                        
            thickness = 1

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


            elif shape_type == 'circle':
                center = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                radius = random.randint(1, min(height, width) // 4)
                cv.circle(image_with_shapes, center, radius, color, thickness)

            elif shape_type == 'gaussian_noise': 
                noise = np.random.normal(0, 5, image_with_shapes.shape).astype(np.uint8)  # Mean=0, Stddev=5
                image_with_shapes = image_with_shapes + 0.001 * noise


            elif shape_type == 'color_blending':
                blend_color = np.full(image_with_shapes.shape, 
                                    (random.randint(0, 255), 
                                    random.randint(0, 255), 
                                    random.randint(0, 255)), 
                                    dtype=np.uint8)  

                alpha = random.uniform(0.1, 0.5) 
                image_with_shapes = (1 - alpha) * image_with_shapes + alpha * blend_color
            
        return image_with_shapes.astype(np.uint8)
        
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