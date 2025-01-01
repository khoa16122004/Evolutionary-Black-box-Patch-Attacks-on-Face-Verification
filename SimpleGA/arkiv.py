import pickle as pkl

with open("arkiv.pkl", "rb") as f:
    arkiv = pkl.load(f)
    
    
print(len(arkiv))
    
