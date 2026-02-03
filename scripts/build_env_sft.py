import json, re, random
from pathlib import Path

random.seed(7)

inp = Path("data/corpus/openalex_corpus.jsonl")
out = Path("data/train/env_sft.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

# 变量关键词：覆盖你们论文叙事里的“参数化暗发酵→产氢”
KEYS = [
  "pH", "HRT", "OLR", "C/N", "temperature", "thermophilic", "mesophilic",
  "volatile fatty", "VFA", "ammonia", "inhibition", "Clostridium", "Enterobacter",
  "pretreatment", "heat shock", "acid treatment", "alkaline", "ultrasound", "microwave",
  "hydrogen yield", "H2", "biohydrogen", "dark fermentation", "food waste", "OFMSW"
]

def pick_spans(text, max_spans=3):
    # 取包含关键词的句段（粗粒度，后续可替换成更强抽取）
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    hits = []
    for s in sents:
        if any(k.lower() in s.lower() for k in KEYS) and 80 <= len(s) <= 400:
            hits.append(s.strip())
    random.shuffle(hits)
    return hits[:max_spans]

def make_qa(span):
    # 三类题型：extractive / range / counterfactual（先弱监督版本）
    t = span
    q_types = [
      ("extract", "From the evidence, extract the key operating parameters and reported outcomes (keep units if present)."),
      ("range", "Based on the evidence, summarize a plausible recommended range/setting for the mentioned parameters and justify briefly using the text."),
      ("counterfactual", "Consider increasing pH by 0.5 within the mentioned context. Predict the likely direction of hydrogen performance and give a mechanistic rationale grounded in the evidence.")
    ]
    qt = random.choice(q_types)
    prompt = (
      "You are EnvironmentGPT, a closed-book environmental bioprocess expert.\n"
      "Task: answer based on scientific evidence you have internalized.\n"
      "Instruction: " + qt[1] + "\n"
      "Evidence:\n" + t + "\n"
      "Answer:"
    )
    # 训练阶段先用“证据式回答”做对齐；推理评测时不提供 Evidence（闭卷）。
    # 这里 response 先用 span 本身做弱监督（extractive），后续你们可替换为 LLM-assisted structured extraction。
    if qt[0] == "extract":
        response = "Key evidence excerpt:\n" + t
    else:
        response = "Evidence-based summary:\n" + t
    return prompt, response

N_MAX = 60000  # 先生成到 6 万条，足够 12–24h LoRA+TopoGuard 训练
count = 0
with inp.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for line in f:
        o = json.loads(line)
        text = o.get("text","")
        spans = pick_spans(text, max_spans=3)
        for sp in spans:
            p, r = make_qa(sp)
            w.write(json.dumps({"type":"env","prompt":p,"response":r}, ensure_ascii=False)+"\n")
            count += 1
            if count >= N_MAX:
                break
        if count >= N_MAX:
            break

print(f"[env_sft] samples={count} out={out}")
