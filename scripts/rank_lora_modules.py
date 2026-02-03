import os, argparse
import pandas as pd
import torch
from safetensors.torch import load_file

def key_to_prefix(k: str, tag: str):
    # 统一匹配：
    #   xxx.lora_A.weight
    #   xxx.lora_A.default.weight
    #   xxx.lora_A.<adapter_name>.weight
    if tag not in k:
        return None
    if not k.endswith(".weight"):
        return None
    return k.split(tag, 1)[0].rstrip(".")

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
        pa = key_to_prefix(k, ".lora_A")
        if pa is not None:
            A[pa] = v
            continue
        pb = key_to_prefix(k, ".lora_B")
        if pb is not None:
            B[pb] = v
            continue

        # embedding LoRA（若存在）
        pea = key_to_prefix(k, ".lora_embedding_A")
        if pea is not None:
            A[pea + ".__embed__"] = v
            continue
        peb = key_to_prefix(k, ".lora_embedding_B")
        if peb is not None:
            B[peb + ".__embed__"] = v
            continue

    mods = sorted(set(A.keys()) & set(B.keys()))
    print(f"[dbg] found A={len(A)} B={len(B)} pairs={len(mods)}")

    if not mods:
        cand = [k for k in sd.keys() if "lora" in k.lower()][:80]
        print("[dbg] sample lora-like keys:")
        for k in cand:
            print(" ", k)
        raise SystemExit("[ERR] no lora_A/lora_B pairs found (adapter key naming unexpected)")

    rows = []
    for m in mods:
        a = A[m].float()
        b = B[m].float()
        # proxy for ||B@A||_F ：||A||_F * ||B||_F（足够做排序 + 稀疏曲线）
        score = (torch.norm(a) * torch.norm(b)).item()
        rows.append({"module": m, "score": score})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.to_csv(args.out_csv, index=False)
    print("[OK] wrote", args.out_csv, "rows=", len(df))

if __name__ == "__main__":
    main()
