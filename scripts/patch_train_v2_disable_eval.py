import pathlib, shutil, re

p = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py")
bak = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py.bak_disable_eval")
shutil.copy2(p, bak)
print("Backup ->", bak)

s = p.read_text(encoding="utf-8")

# 1) eval_strategy -> "no"
if "eval_strategy" in s:
    s2, n = re.subn(r"(eval_strategy\s*=\s*)(['\"]).*?\2", r"\1'no'", s, count=1)
    s = s2
    print("Patched eval_strategy -> 'no' (replaced:", n, ")")
else:
    print("WARNING: eval_strategy not found; will try to inject later (not implemented).")

# 2) 强制 per_device_eval_batch_size=1（即使未来开启 eval 也更安全）
if "per_device_eval_batch_size" not in s:
    s, n = re.subn(
        r"(TrainingArguments\s*\(\s*)",
        r"\1\n        per_device_eval_batch_size=1,\n",
        s,
        count=1
    )
    print("Injected per_device_eval_batch_size=1 (inserted:", n, ")")
else:
    s2, n = re.subn(r"(per_device_eval_batch_size\s*=\s*)\d+", r"\g<1>1", s, count=1)
    s = s2
    print("Set per_device_eval_batch_size=1 (replaced:", n, ")")

p.write_text(s, encoding="utf-8")
print("Patched file ->", p)
