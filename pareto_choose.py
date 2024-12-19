merge_arkive  = sum(arkves)
merge_arkive.sort(key=lambda x: (x['pnsr_score'], x['adv_score']), reverse=True)

final_arkive = []

for item in merge_arkive:
    final_arkive = arkive_proccesing(final_arkive, merge_arkive)
    