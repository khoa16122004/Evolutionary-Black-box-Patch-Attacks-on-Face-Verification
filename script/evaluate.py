import os
import pickle

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def eval(path):
    data = read_pickle(path)
    adv_score = [d["best_adv_score"] for d in data]
    success_cnt = 0
    for d in data:
        if d["best_adv_score"] > 0:
            success_cnt += 1
    psnr = [d["best_psnr"] for d in data]
    return success_cnt, sum(adv_score)/len(adv_score), sum(psnr)/len(psnr)
def find_best(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    data = data.split(":")[1].split(",")[0].split()[:2]
    result = [float(value) for value in data]
    return result[0], result[1], result[0]>0
if __name__ == "__main__":
    path = r"D:\codePJ\RESEARCH\GECCO2025\results_final\BF_results\image-22\best_adv_22.txt"
    # cnt, adv_score, psnr = eval(path)
    # print(f"Success Rate: {cnt}/100 , Mean Adversarial Score: {adv_score}, Mean PSNR: {psnr}")
    # f.write(f"{self.idx}: {best_psnr_adv['adv_score']} {best_psnr_adv['psnr']}, {best_adv['adv_score']} {best_adv['psnr']}\n")
    sum_adv, sum_psnr, cnt = 0, 0, 0
    for i in range(0, 100):
        path = f"D:\\codePJ\\RESEARCH\\GECCO2025\\results_final\\BF_results\\image-{i}\\best_adv_{i}.txt"
        adv_score, psnr, success = find_best(path)
        sum_adv += adv_score
        sum_psnr += psnr
        if success:
            cnt += 1
    print(f"Mean Adversarial Score: {sum_adv/100}, Mean PSNR: {sum_psnr/100}, Success Rate: {cnt}/100")
    