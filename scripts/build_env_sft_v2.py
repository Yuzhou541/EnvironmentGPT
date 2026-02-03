import json, re, random, hashlib
from pathlib import Path

random.seed(7)

inp = Path("data/corpus/openalex_corpus.jsonl")
out = Path("data/train/env_sft.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

KEYS = [
  "pH", "HRT", "OLR", "C/N", "temperature", "thermophilic", "mesophilic",
  "volatile fatty", "VFA", "ammonia", "inhibition", "methanogen",
  "Clostridium", "Enterobacter", "pretreatment", "heat shock", "acid treatment",
  "alkaline", "enzymatic", "ultrasound", "microwave", "steam explosion",
  "hydrogen yield", "H2", "biohydrogen", "dark fermentation", "food waste", "OFMSW"
]

def normalize_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def chunk_text(t: str, win=1200, step=500):
    # 字符滑窗切块，避免 PDF 没句号导致抓不到
    n = len(t)
    for s in range(0, max(1, n - 1), step):
        c = t[s:s+win]
        if len(c) < 200:
            continue
        yield c

def hit(c: str) -> bool:
    cl = c.lower()
    return any(k.lower() in cl for k in KEYS)

def make_qa(span: str):
    t = span.strip()
    q_types = [
      ("extract", "Extract the key operating parameters and reported outcomes (keep units if present)."),
      ("range", "Summarize plausible recommended ranges/settings for the mentioned parameters and justify briefly."),
      ("counterfactual", "If pH increases by 0.5 under the described context, predict the likely direction of hydrogen performance and give a brief mechanistic rationale.")
    ]
    qt = random.choice(q_types)
    prompt = (
      "You are EnvironmentGPT, a closed-book environmental bioprocess expert.\n"
      "Task: answer based on scientific evidence you have internalized.\n"
      f"Instruction: {qt[1]}\n"
      f"Evidence:\n{t}\n"
      "Answer:"
    )
    if qt[0] == "extract":
        response = "Key evidence excerpt:\n" + t
    else:
        response = "Evidence-based summary:\n" + t
    return prompt, response

N_MAX = 80000     # 建议 8 万起步；你要更久训练可改 120000
MAX_PER_PDF = 80  # 每篇最多取多少块（防止少数超长 PDF 独占）
seen = set()

count = 0
pdf_ok = 0
with inp.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for line in f:
        o = json.loads(line)
        text = normalize_text(o.get("text",""))
        got = 0
        for c in chunk_text(text):
            if not hit(c):
                continue
            h = hashlib.md5(c.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            p, r = make_qa(c)
            w.write(json.dumps({"type":"env","prompt":p,"response":r}, ensure_ascii=False) + "\n")
            count += 1
            got += 1
            if got >= MAX_PER_PDF or count >= N_MAX:
                break
        if got > 0:
            pdf_ok += 1
        if count >= N_MAX:
            break

print(f"[env_sft_v2] samples={count}, pdf_with_hits={pdf_ok}, out={out}")
