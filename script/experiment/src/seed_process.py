import os
import numpy as np

# Danh sách các thư mục seed
seed_folders = [
    "seed_22520691",
    "seed_22520692",
    "seed_22520693",
    "seed_22520694",
    "seed_22520695"
]

# Các phương pháp trong mỗi thư mục
methods = ["adaptive", "normal", "nsgaii", "rules"]

# Tạo thư mục để lưu kết quả
save_folder = "mean_across_seeds"
os.makedirs(save_folder, exist_ok=True)

# Duyệt qua từng phương pháp
for method in methods:
    adv_scores = []
    psnr_scores = []

    # Duyệt qua từng thư mục seed để lấy dữ liệu
    for seed in seed_folders:
        adv_path = os.path.join(seed, f"{method}_adv.txt")
        psnr_path = os.path.join(seed, f"{method}_psnr.txt")

        if os.path.exists(adv_path) and os.path.exists(psnr_path):
            adv_scores.append(np.loadtxt(adv_path))
            psnr_scores.append(np.loadtxt(psnr_path))

    # Chuyển thành mảng numpy
    adv_scores = np.array(adv_scores)  # Shape: (5, 10000)
    psnr_scores = np.array(psnr_scores)  # Shape: (5, 10000)

    # Tính trung bình theo từng iteration
    mean_adv = np.mean(adv_scores, axis=0)
    mean_psnr = np.mean(psnr_scores, axis=0)

    # Lưu kết quả
    np.savetxt(os.path.join(save_folder, f"mean_adv_{method}.txt"), mean_adv, fmt="%.6f")
    np.savetxt(os.path.join(save_folder, f"mean_psnr_{method}.txt"), mean_psnr, fmt="%.6f")

print(f"Đã lưu kết quả vào thư mục: {save_folder}")
