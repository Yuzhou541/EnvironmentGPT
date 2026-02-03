import pathlib, shutil, re

p = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py")
bak = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py.bak_seed_eval_safe")
shutil.copy2(p, bak)
print("Backup ->", bak)

lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)
txt = "".join(lines)

# 如果已经有我们的 patch 标记，就不重复打
if "# [patch] deterministic seed + eval strategy knobs" in txt:
    print("Already patched, skip.")
    raise SystemExit(0)

# --- 找到“import 区域结束”的位置（支持 from ... import ( ... ) 这种括号多行） ---
i = 0
n = len(lines)

# 跳过 shebang / encoding / 空行 / 注释
while i < n and (lines[i].startswith("#!") or re.match(r"^#.*coding[:=]", lines[i]) or lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
    i += 1

in_paren = 0
while i < n:
    ln = lines[i]
    stripped = ln.strip()

    # import 行
    if stripped.startswith("import ") or stripped.startswith("from "):
        # 统计括号深度：处理 from transformers import ( ... )
        in_paren += ln.count("(") - ln.count(")")
        i += 1
        # 若进入括号 import 块，直到括号闭合为止都属于 import 区
        while i < n and in_paren > 0:
            ln2 = lines[i]
            in_paren += ln2.count("(") - ln2.count(")")
            i += 1
        continue

    # 空行也算 import 区（很多人 import 后空一行）
    if stripped == "":
        i += 1
        continue

    # 遇到第一个“非 import/非空行/非注释”语句 => import 区结束
    break

insert_at = i

inject = [
    "\n",
    "# [patch] deterministic seed + eval strategy knobs\n",
    "from transformers import set_seed\n",
    "SEED = int(os.environ.get('SEED', '0'))\n",
    "set_seed(SEED)\n",
    "EVAL_STRATEGY = os.environ.get('EVAL_STRATEGY', 'no')\n",
    "\n",
]

# 确保 import os 存在（你的脚本里一般有，但这里做兜底）
if not re.search(r"^\s*import\s+os(\s|$)", txt, flags=re.M):
    # 若没 import os，就在 inject 前额外加
    inject = ["import os\n"] + inject

lines = lines[:insert_at] + inject + lines[insert_at:]
txt = "".join(lines)

# 兼容：evaluation_strategy -> eval_strategy
txt = txt.replace("evaluation_strategy=", "eval_strategy=")

# 如果 eval_strategy 不是 EVAL_STRATEGY，强制替换为 EVAL_STRATEGY
txt = re.sub(r"eval_strategy\s*=\s*['\"].*?['\"]\s*,", "eval_strategy=EVAL_STRATEGY,", txt)

# 训练/评测省显存：强制 prediction_loss_only=True（如果 TrainingArguments 里还没有）
if "prediction_loss_only=" not in txt:
    m = re.search(r"(TrainingArguments\s*\(\s*\n)", txt)
    if m:
        idx = m.end(1)
        txt = txt[:idx] + "        prediction_loss_only=True,\n" + txt[idx:]

p.write_text(txt, encoding="utf-8")
print("Patched ->", p)
