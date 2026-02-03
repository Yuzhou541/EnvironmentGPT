import argparse, os
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_rank_csv", required=True)
    ap.add_argument("--out_rank_csv", required=True)
    ap.add_argument("--rand_seed", type=int, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_rank_csv)
    if "module" not in df.columns:
        raise SystemExit("[ERR] in_rank_csv must contain column: module")

    rng = np.random.default_rng(args.rand_seed)
    df = df.copy()
    df["score"] = rng.random(len(df))  # random scores => random top-k
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    os.makedirs(os.path.dirname(args.out_rank_csv), exist_ok=True)
    df.to_csv(args.out_rank_csv, index=False)
    print("[OK] wrote", args.out_rank_csv, "rows=", len(df))

if __name__ == "__main__":
    main()
