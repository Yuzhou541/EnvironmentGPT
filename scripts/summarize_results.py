import glob, json, re, os
import pandas as pd

R = "/root/EnvironmentGPT/results"

def read_one(p):
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "summary" in obj and isinstance(obj["summary"], dict):
        return obj["summary"]
    return obj

# -------- raw per-seed (dedup) --------
# Prefer env_table_seedX_raw.json. If missing, fallback to env_table_seedX.json.
seed_to_path = {}

# 1) raw first
for p in sorted(glob.glob(os.path.join(R, "env_table_seed*_raw.json"))):
    bn = os.path.basename(p)
    m = re.match(r"env_table_seed(\d+)_raw\.json$", bn)
    if m:
        seed_to_path[int(m.group(1))] = p

# 2) fallback non-raw (only if raw missing)
for p in sorted(glob.glob(os.path.join(R, "env_table_seed*.json"))):
    bn = os.path.basename(p)
    if "pruned_" in bn or bn.endswith("_raw.json"):
        continue
    m = re.match(r"env_table_seed(\d+)\.json$", bn)
    if m:
        seed = int(m.group(1))
        if seed not in seed_to_path:
            seed_to_path[seed] = p

if not seed_to_path:
    raise SystemExit("[ERR] no env_table_seed*.json found")

rows = []
for seed in sorted(seed_to_path.keys()):
    s = read_one(seed_to_path[seed])
    rows.append({"seed": seed, **s})

df = pd.DataFrame(rows).sort_values("seed")
out_csv = os.path.join(R, "summary_env_table_seeds_raw.csv")
df.to_csv(out_csv, index=False)

num_cols = [c for c in df.columns if c != "seed"]
mean = df[num_cols].mean(numeric_only=True)
std  = df[num_cols].std(numeric_only=True)
ms = pd.DataFrame([{"metric": k, "mean": float(mean[k]), "std": float(std[k])} for k in mean.index])
out_ms = os.path.join(R, "summary_env_table_seeds_raw_mean_std.csv")
ms.to_csv(out_ms, index=False)

# -------- pruning curve --------
prows = []
for p in sorted(glob.glob(os.path.join(R, "env_table_seed*_pruned_*.json"))):
    bn = os.path.basename(p)
    m = re.match(r"env_table_seed(\d+)_pruned_([0-9.]+)\.json$", bn)
    if not m:
        continue
    seed = int(m.group(1))
    k = float(m.group(2))
    s = read_one(p)
    prows.append({"seed": seed, "keep_frac": k, **s})

out_pcsv = os.path.join(R, "summary_pruning_curve.csv")
if prows:
    pdf = pd.DataFrame(prows).sort_values(["seed", "keep_frac"])
    pdf.to_csv(out_pcsv, index=False)
else:
    pd.DataFrame(columns=["seed","keep_frac"]).to_csv(out_pcsv, index=False)

print("[OK] wrote:")
print(" -", out_csv)
print(" -", out_ms)
print(" -", out_pcsv)
print("[OK] raw used:")
for seed in sorted(seed_to_path):
    print("  seed", seed, "->", os.path.basename(seed_to_path[seed]))
