import os
import numpy as np
acc = 0
psnr = 0
def argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])

def ruled_selection(iter_adv_scores, iter_psnr_scores):
    success_indexs = []
    for k in range(len(iter_adv_scores)):
        if iter_adv_scores[k] >= 0:
            success_indexs.append(k)

    if len(success_indexs) > 0: # if exist successfully
        iter_success_psnr_scores = [iter_psnr_scores[i] for i in success_indexs] # pnsr of success
        iter_success_adv_scores = [iter_adv_scores[i] for i in success_indexs] # adv of success
        
        best_psnr_iter_success = argmax(iter_success_psnr_scores)
        return iter_success_adv_scores[best_psnr_iter_success], iter_success_psnr_scores[best_psnr_iter_success], True
    
    else:
        best_adv_iter = argmax(iter_adv_scores)
        return iter_adv_scores[best_adv_iter], iter_psnr_scores[best_adv_iter], False


result_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520693_final_NSGAII_selected"

for i in range(0, 100):
    result_path = os.path.join(result_dir, f"{i}.txt")
    with open(result_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        adv_scores = [float(line[0]) for line in lines]
        psnr_scores = [float(line[1]) for line in lines]
        adv_score, psnr_score = ruled_selection(adv_scores, psnr_scores)
        if float(adv_score) >= 0:
            acc += 1
        psnr += float(psnr_score)
        
print(acc / 100)
print(psnr / 100)
        
        