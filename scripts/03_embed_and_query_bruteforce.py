import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

chunks_path = Path("data/processed/chunks_q1.jsonl")
index_dir = Path("data/index_bruteforce_q1")
index_dir.mkdir(parents=True, exist_ok=True)

# 1) 读 chunks
texts = []
meta = []
with chunks_path.open("r", encoding="utf-8") as r:
    for line in r:
        rec = json.loads(line)
        texts.append(rec["text"])
        meta.append({"chunk_id": rec["chunk_id"], "pdf_file": rec["pdf_file"]})

if len(texts) == 0:
    raise RuntimeError("chunks_q1.jsonl is empty. Check corpus extraction / chunking.")

print("chunks =", len(texts))

# 2) 编码（第一次会从 HuggingFace 拉模型）
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

emb = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
).astype(np.float32)  # 已归一化 -> cosine 相似度 = 点积

np.save(index_dir / "emb.npy", emb)
with (index_dir / "meta.json").open("w", encoding="utf-8") as w:
    json.dump({"model": model_name, "meta": meta}, w, ensure_ascii=False)

# 3) TopK 查询（暴力：emb @ qemb）
q = "optimal pH and HRT for dark fermentation biohydrogen from food waste"
qemb = model.encode([q], normalize_embeddings=True).astype(np.float32)[0]

sims = emb @ qemb  # (N,)
topk = 5
idx = np.argpartition(-sims, kth=min(topk, len(sims)-1))[:topk]
idx = idx[np.argsort(-sims[idx])]

print("\nQuery:", q)
print("Top-5:")
for rank, i in enumerate(idx, 1):
    m = meta[int(i)]
    print(f"{rank}. sim={float(sims[i]):.4f}  pdf={m['pdf_file']}  chunk={m['chunk_id']}")
