import pandas as pd
f = pd.read_csv("results/ref_window_sweep_folds.csv")
p = f[f["eval"]=="fixed"].pivot(index="subject", columns="n_ref", values="balanced_acc")
print(p.round(3).sort_values(10))