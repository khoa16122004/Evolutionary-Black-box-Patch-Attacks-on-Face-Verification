import os

acc = 0
psnr = 0
result_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\22520695_final_rules_selected" 

for i in range(0, 100):
    result_path = os.path.join(result_dir, f"final_selected_{i}.txt")
    with open(result_path, "r") as f:
        line = f.readline()
        [adv_scores, psnr_scores] = line.strip().split()        
        if float(adv_scores) >= 0:
            acc += 1
        psnr += float(psnr_scores)
        
print("SSR: ", acc / 100)
print("PSNR: ", psnr / 100)