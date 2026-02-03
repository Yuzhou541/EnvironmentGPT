import re, shutil, pathlib

path = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py")
bak  = pathlib.Path("/root/EnvironmentGPT/scripts/train_lora_sft_v2.py.bak_textfield")
shutil.copy2(path, bak)
print("Backup ->", bak)

lines = path.read_text(encoding="utf-8").splitlines(True)

# 1) find tokenize_fn block start
start = None
for i, ln in enumerate(lines):
    if re.match(r'^\s*def\s+tokenize_fn\s*\(\s*examples\s*\)\s*:\s*$', ln):
        start = i
        break
if start is None:
    raise SystemExit("Cannot find def tokenize_fn(examples):")

# 2) find the next top-level statement after tokenize_fn (heuristic: first line that starts with 'dataset = dataset.map' or 'dataset = dataset[')
end = None
for j in range(start+1, len(lines)):
    if re.match(r'^\s*dataset\s*=\s*dataset\.map\s*\(', lines[j]) or re.match(r'^\s*dataset\s*=\s*dataset\[[\'"]', lines[j]):
        end = j
        break
if end is None:
    raise SystemExit("Cannot find end of tokenize_fn block (anchor not found).")

new_block = r'''
def _build_texts(examples, tokenizer):
    # batched=True: examples is dict[str, list]
    if "text" in examples:
        return examples["text"]

    # chat-style
    if "messages" in examples:
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
    if "conversations" in examples:
        # some datasets use "conversations" instead of "messages"
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["conversations"]
        ]

    # prompt/response style
    if "prompt" in examples and "response" in examples:
        return [
            tokenizer.apply_chat_template(
                [{"role":"user","content":p},{"role":"assistant","content":r}],
                tokenize=False,
                add_generation_prompt=False
            )
            for p, r in zip(examples["prompt"], examples["response"])
        ]

    # alpaca-style
    if "instruction" in examples and "output" in examples:
        inputs = examples.get("input", [""] * len(examples["instruction"]))
        texts = []
        for inst, inp, out in zip(examples["instruction"], inputs, examples["output"]):
            u = inst if not inp else (inst + "\n" + inp)
            texts.append(
                tokenizer.apply_chat_template(
                    [{"role":"user","content":u},{"role":"assistant","content":out}],
                    tokenize=False,
                    add_generation_prompt=False
                )
            )
        return texts

    raise KeyError(f"Unsupported dataset columns: {list(examples.keys())}")

def tokenize_fn(examples):
    texts = _build_texts(examples, tokenizer)
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
    )

'''
# preserve indentation at column 0
new_lines = [ln if ln.endswith("\n") else ln + "\n" for ln in new_block.splitlines(True)]

out = lines[:start] + new_lines + lines[end:]
path.write_text("".join(out), encoding="utf-8")
print("Patched:", path)
