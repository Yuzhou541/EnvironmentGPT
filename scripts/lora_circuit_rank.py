import os, json, argparse
import torch
from safetensors.torch import load_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_csv", default="results/lora_circuit_rank.csv")
    args = ap.parse_args()

    st_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")
    sd = load_file(st_path)

    # keys look like: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
    pairs = {}
    for k, v in sd.items():
        if ".lora_A." in k:
            base = k.replace(".lora_A.weight", "")
            pairs.setdefault(base, {})["A"] = v
        elif ".lora_B." in k:
            base = k.replace(".lora_B.weight", "")
            pairs.setdefault(base, {})["B"] = v

    rows = []
    for base, d in pairs.items():
        A = d.get("A", None)
        B = d.get("B", None)
        if A is None or B is None:
            continue
        # proxy circuit strength: ||B||_F * ||A||_F (scale-invariant-ish)
        s = (torch.norm(A.float(), p="fro") * torch.norm(B.float(), p="fro")).item()
        rows.append((base, s, tuple(A.shape), tuple(B.shape)))

    rows.sort(key=lambda x: x[1], reverse=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("module,score,A_shape,B_shape\n")
        for m, s, a, b in rows:
            f.write(f"{m},{s:.6g},{a},{b}\n")

    print(f"Wrote {len(rows)} modules to {args.out_csv}")
    print("Top-10:")
    for m, s, *_ in rows[:10]:
        print(f"{s:.6g}\t{m}")

if __name__ == "__main__":
    main()
