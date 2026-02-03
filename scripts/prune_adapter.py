import os, argparse, shutil
import torch
from safetensors.torch import load_file, save_file

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--rank_csv", required=True)
    ap.add_argument("--keep_frac", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    args=ap.parse_args()

    # read module ranking
    modules=[]
    with open(args.rank_csv,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
            modules.append(line.split(",",1)[0])
    keep_n=max(1, int(len(modules)*args.keep_frac))
    keep=set(modules[:keep_n])

    # copy adapter_dir (config etc.)
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for fn in os.listdir(args.adapter_dir):
        if fn.endswith(".safetensors"):
            continue
        src=os.path.join(args.adapter_dir, fn)
        dst=os.path.join(args.out_dir, fn)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # prune safetensors
    st_path=os.path.join(args.adapter_dir, "adapter_model.safetensors")
    sd=load_file(st_path)
    new={}
    for k,v in sd.items():
        if ".lora_A." in k:
            base=k.replace(".lora_A.weight","")
            if base not in keep: v=torch.zeros_like(v)
        elif ".lora_B." in k:
            base=k.replace(".lora_B.weight","")
            if base not in keep: v=torch.zeros_like(v)
        new[k]=v
    save_file(new, os.path.join(args.out_dir, "adapter_model.safetensors"))
    print(f"keep_frac={args.keep_frac} kept_modules={keep_n}/{len(modules)} -> {args.out_dir}")

if __name__=="__main__":
    main()
