import re, shutil, pathlib

p = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py")
bak = p.with_suffix(".py.bak_seed_eval")
shutil.copy2(p, bak)
print("Backup ->", bak)

s = p.read_text(encoding="utf-8").splitlines(True)

# 1) ensure set_seed + SEED injected near imports
txt = "".join(s)
if "set_seed(" not in txt:
    # find first "from transformers" import line or after imports
    insert_at = None
    for i, ln in enumerate(s):
        if re.match(r'^\s*from\s+transformers\s+import\s+', ln):
            insert_at = i + 1
            break
    if insert_at is None:
        # after imports block
        for i, ln in enumerate(s):
            if re.match(r'^\s*import\s+', ln):
                insert_at = i + 1
        if insert_at is None:
            insert_at = 0

    inject = [
        "\n",
        "# [patch] deterministic seed + eval strategy knobs\n",
        "from transformers import set_seed\n",
        "SEED = int(os.environ.get('SEED', '0'))\n",
        "set_seed(SEED)\n",
        "EVAL_STRATEGY = os.environ.get('EVAL_STRATEGY', 'no')\n",
        "\n",
    ]
    s = s[:insert_at] + inject + s[insert_at:]

# 2) TrainingArguments: evaluation_strategy <-> eval_strategy compatibility
txt = "".join(s)
# if it uses evaluation_strategy, replace to eval_strategy (your env seems to require eval_strategy)
txt = txt.replace("evaluation_strategy=", "eval_strategy=")

# 3) force prediction_loss_only to reduce eval memory if eval is ever enabled
if "prediction_loss_only=" not in txt:
    # inject into TrainingArguments(...) call
    # find "TrainingArguments(" and inject right after it
    pattern = r"(TrainingArguments\s*\(\s*\n)"
    m = re.search(pattern, txt)
    if m:
        idx = m.end(1)
        txt = txt[:idx] + "        prediction_loss_only=True,\n" + txt[idx:]

# 4) if eval_strategy exists in TrainingArguments call but set to something else, replace with EVAL_STRATEGY
txt = re.sub(r"eval_strategy\s*=\s*['\"].*?['\"]\s*,", "eval_strategy=EVAL_STRATEGY,", txt)

p.write_text(txt, encoding="utf-8")
print("Patched ->", p)
