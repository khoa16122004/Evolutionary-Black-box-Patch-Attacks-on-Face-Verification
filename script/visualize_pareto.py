import matplotlib.pyplot as plt

with open(r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520691_NSGAII_arkv\0.txt", "r") as f:
    lines = [line.strip().split() for line in f]
    points = [(float(line[0]), float(line[1])) for line in lines]

points = sorted(points, key=lambda x: x[0])  # Sắp xếp theo adv_scores

adv_scores, psnr_scores = zip(*points)

plt.figure(figsize=(8, 6))
plt.plot(adv_scores, psnr_scores, color='blue', marker='o', label='Pareto Front', alpha=0.7)

plt.title('Pareto Front: Adv Scores vs PSNR Scores', fontsize=14)
plt.xlabel('Adversarial Scores', fontsize=12)
plt.ylabel('PSNR Scores', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.show()
