import numpy as np

# Giả sử có hai mảng
arr1 = [0.22, 0.18,0.17 ,0.16 , 0.17 ]
arr2 = [0.2515, 0.2713, 0.2712, 0.2712,0.2713]

mean_arr1 = np.mean(arr1)
variance_arr1 = np.var(arr1)
std_arr1 = np.std(arr1)

mean_arr2 = np.mean(arr2)
variance_arr2 = np.var(arr2)
std_arr2 = np.std(arr2)

print(f"Mean of arr1: {mean_arr1}, Variance of arr1: {variance_arr1}, Standard Deviation of arr1: {std_arr1}")
print(f"Mean of arr2: {mean_arr2}, Variance of arr2: {variance_arr2}, Standard Deviation of arr2: {std_arr2}")
