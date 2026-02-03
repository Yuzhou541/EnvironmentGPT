import pathlib, shutil, re

p = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py")
bak = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py.bak_maxseqlen")
shutil.copy2(p, bak)
print("Backup ->", bak)

s = p.read_text(encoding="utf-8").splitlines(True)

# 如果已经有全局 max_seq_len 就不重复注入
if any(re.match(r'^\s*max_seq_len\s*=', ln) for ln in s):
    print("Found existing global max_seq_len, skip inject.")
    raise SystemExit(0)

# 找到 import os 的位置，在其后插入 max_seq_len 定义
idx = None
for i, ln in enumerate(s):
    if re.match(r'^\s*import\s+os(\s|$)', ln) or re.match(r'^\s*from\s+os\s+import\s+', ln):
        idx = i
        break

inject = [
    "\n",
    "# [patch] ensure global max_seq_len for tokenize_fn()\n",
    "max_seq_len = int(os.environ.get('MAX_SEQ_LEN', os.environ.get('MAX_SEQ_LENGTH', '2048')))\n",
    "\n",
]

if idx is None:
    # 没找到 import os，就在文件最开头插入 import os + 定义
    s = ["import os\n"] + inject + s
    print("Injected import os + max_seq_len at file top.")
else:
    s = s[:idx+1] + inject + s[idx+1:]
    print("Injected max_seq_len after import os.")

p.write_text("".join(s), encoding="utf-8")
print("Patched:", p)
