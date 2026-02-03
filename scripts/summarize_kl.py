import glob, json, os, re
import pandas as pd

R="/root/EnvironmentGPT/results"
rows=[]
for p in sorted(glob.glob(os.path.join(R, "kl_seed*_anchor300_topk50.json"))):
    m=re.search(r"kl_seed(\d+)_anchor300_topk50\.json$", os.path.basename(p))
    if not m: 
        continue
    seed=int(m.group(1))
    with open(p,"r",encoding="utf-8") as f:
        obj=json.load(f)
    rows.append({"seed":seed, **obj})

df=pd.DataFrame(rows).sort_values("seed")
out=os.path.join(R, "summary_kl_anchor300_topk50.csv")
df.to_csv(out, index=False)
print("[OK] wrote", out)
print(df)
