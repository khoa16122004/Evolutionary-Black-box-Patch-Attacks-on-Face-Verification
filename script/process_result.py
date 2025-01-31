import os
from tqdm import tqdm
import pickle as pkl
import argparse

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
        return iter_success_adv_scores[best_psnr_iter_success], iter_success_psnr_scores[best_psnr_iter_success]
    
    else:
        best_adv_iter = argmax(iter_adv_scores)
        return iter_adv_scores[best_adv_iter], iter_psnr_scores[best_adv_iter]
    
def take_data(pkl_file, algorithm):
    # return log and return final Population
    population = pkl_file['Population']
    
    final_adv_scores = [ind.adv_score for ind in population]
    final_psnr_scores = [ind.psnr_score for ind in population]

    final_selected_adv, final_selected_psnr = ruled_selection(final_adv_scores, final_psnr_scores)
    
    log = pkl_file['log']
    adv_scores_log = [] # will returned
    psnr_scores_log = [] # will returned
    
    if algorithm.startswith("GA"):
        print("Proccessing ", algorithm)
        for i in range(0, len(log), 2):
            iter_log = log[i: i + 2]
            
            iter_adv_scores = iter_log[0]['adv_scores_log'] + iter_log[1]['adv_scores_log']
            iter_psnr_scores = iter_log[0]['psnr_scores_log'] + iter_log[1]['psnr_scores_log']

            
            if algorithm == "GA_normal":
                current_combined_fitnesses = [
                    0.5 * iter_adv_scores[j] + 0.5 * iter_psnr_scores[j] 
                    for j in range(len(iter_adv_scores))
                ]
                
                best_combined_fitnesse_index = argmax(current_combined_fitnesses)
                selected_adv_score, selected_psnr_score = iter_adv_scores[best_combined_fitnesse_index], iter_psnr_scores[best_combined_fitnesse_index]
                    
            elif algorithm == "GA_adaptive":
                current_combined_fitnesses = [
                    0.5 * iter_psnr_scores[j] if iter_adv_scores[j] >= 0 
                    else 0.5 * iter_adv_scores[j] + 0.5 * iter_psnr_scores[j] 
                    for j in range(len(iter_adv_scores))
                ]
                
                best_combined_fitnesse_index = argmax(current_combined_fitnesses)
                selected_adv_score, selected_psnr_score = iter_adv_scores[best_combined_fitnesse_index], iter_psnr_scores[best_combined_fitnesse_index]
            
            else: # GA rule base and NSGAII
                # take with rule 
                selected_adv_score, selected_psnr_score = ruled_selection(iter_adv_scores, iter_psnr_scores)               
            
            adv_scores_log.append(selected_adv_score)
            psnr_scores_log.append(selected_psnr_score)           
    else:
        print("Proccessing NSGAII")
        for i in range(len(log)):
            iter_log = log[i]
            adv_scores = iter_log['adv_scores_log']
            psnr_scores = iter_log['psnr_scores_log']   
            
            iter_adv_scores = []
            iter_psnr_scores = []
            for j in range(len(adv_scores)):
                iter_adv_scores.extend(list(adv_scores[j].cpu()))
                iter_psnr_scores.extend(list(psnr_scores[j].cpu()))
                           
            selected_adv_score, selected_psnr_score = ruled_selection(iter_adv_scores, iter_psnr_scores)               
            
            adv_scores_log.append(selected_adv_score)
            psnr_scores_log.append(selected_psnr_score)     
    
    return (adv_scores_log, psnr_scores_log), (final_selected_adv, final_selected_psnr)
            
                    
    

def load_file(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pkl.load(f)
    
def main(args):
    
    os.makedirs(args.output_seleted_dir, exist_ok=True)
    os.makedirs(args.output_final_seleted_dir, exist_ok=True)
    
    for i in tqdm(range(len(os.listdir(args.pkl_dir)))):
        # if i == 0:
        #     continue
        # i = 2
        pkl_path = os.path.join(args.pkl_dir, f"{i}.pkl")
        # pkl_path = r"D:\Path-Recontruction-with-Evolution-Strategy\experiment\src\0.pkl"
        print("Loading file: ", pkl_path)
        try:
            pkl_file = load_file(pkl_path)
        except:
            continue
        (adv_scores_log, psnr_scores_log), (final_selected_adv, final_selected_psnr) = take_data(pkl_file, args.algorithm)
        
        print("Save results")
        # selected_i.txt  
        output_seleted_file = os.path.join(args.output_seleted_dir, f"selected_{i}.txt")      
        output_final_selected_file = os.path.join(args.output_final_seleted_dir, f"final_selected_{i}.txt")
        with open(output_seleted_file, "w") as f:
            for adv_score, psnr_score in zip(adv_scores_log, psnr_scores_log):
                f.write(f"{adv_score} {psnr_score}\n")
        
        # final_selected_i.txt
        with open(output_final_selected_file, "w") as f:
            f.write(f"{final_selected_adv} {final_selected_psnr}\n")
            
        # break

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_dir", type=str, required=True)
    parser.add_argument("--output_seleted_dir", type=str, required=True)
    parser.add_argument("--output_final_seleted_dir", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    args = parser.parse_args()
    main(args)
# python process_result.py     
# --pkl_dir D:\Path-Recontruction-with-Evolution-Strategy\experiment\seed=22520691_arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle
# --output_selected_dir 22520691_adaptive_selected
# --output_final_selected_dir 22520691_adaptive_final_selected
# --algorithm GA_adaptive


# python process_result.py --pkl_dir D:\Path-Recontruction-with-Evolution-Strategy\experiment\seed=22520691_arkiv_GA_rules_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle --output_seleted_dir 22520691_rulebased_selected --output_final_seleted_dir 22520691_adaptive_rulebased_selected --algorithm GA_rulebased
