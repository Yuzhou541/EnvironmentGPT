import os, argparse
import pandas as pd
import torch
from safetensors.torch import load_file

def base_prefix_from_key(k: str, tag: str):
    # tag in {"lora_A", "lora_B"}
    # supports:
    #   xxx.lora_A.weight
    #   xxx.lora_A.default.weight
    #   xxx.lora_A.<any>.weight
    needle = f".{tag}."
    if needle not in k or not k.endswith(".weight"):
        return None
    return k.split(needle, 1)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    st_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(st_path):
        raise SystemExit(f"[ERR] missing {st_path}")

    sd = load_file(st_path)  # CPU tensors
    A, B = {}, {}

    for k, v in sd.items():
        pa = base_prefix_from_key(k, "lora_A")
        if pa is not None:
            A[pa] = v
            continue
        pb = base_prefix_from_key(k, "lora_B")
        if pb is not None:
            B[pb] = v
            continue

    mods = sorted(set(A.keys()) & set(B.keys()))
    if not mods:
        # help debug: show some keys
        sample_keys = list(sd.keys())[:50]
        raise SystemExit("[ERR] no lora_A/lora_B pairs found. Sample keys:\n" + "\n".join(sample_keys))

    rows = []
    for m in mods:
        a = A[m].float()
        b = B[m].float()
        # proxy score for ||B@A||_F : ||A||_F * ||B||_F (fast, stable for ranking)
        score = (torch.norm(a) * torch.norm(b)).item()
        rows.append({"module": m, "score": score})

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"[OK] wrote {args.out_csv}")
    print(f"[OK] modules={len(df)}  top5=")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
