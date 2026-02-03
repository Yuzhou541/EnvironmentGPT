import pathlib, shutil

p = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py")
bak = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py.bak_evalstrategy")
shutil.copy2(p, bak)
print("Backup ->", bak)

s = p.read_text(encoding="utf-8")
if "evaluation_strategy" not in s:
    print("No 'evaluation_strategy' found; nothing to patch.")
else:
    s2 = s.replace("evaluation_strategy", "eval_strategy")
    p.write_text(s2, encoding="utf-8")
    print("Patched: evaluation_strategy -> eval_strategy")
