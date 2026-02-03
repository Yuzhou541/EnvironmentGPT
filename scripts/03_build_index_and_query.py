import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hnswlib
from sentence_transformers import SentenceTransformer

chunks_path = Path("data/processed/chunks_q1.jsonl")
index_dir = Path("data/index_hnsw_q1")
index_dir.mkdir(parents=True, exist_ok=True)

texts = []
meta = []
with chunks_path.open("r", encoding="utf-8") as r:
    for line in r:
        rec = json.loads(line)
        texts.append(rec["text"])
        meta.append({"chunk_id": rec["chunk_id"], "pdf_file": rec["pdf_file"]})

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
emb = np.asarray(emb, dtype=np.float32)

dim = emb.shape[1]
p = hnswlib.Index(space="cosine", dim=dim)
p.init_index(max_elements=emb.shape[0], ef_construction=200, M=16)
p.add_items(emb, np.arange(emb.shape[0]))
p.set_ef(50)

p.save_index(str(index_dir / "hnsw.bin"))
with (index_dir / "meta.json").open("w", encoding="utf-8") as w:
    json.dump(meta, w, ensure_ascii=False)

# Query
q = 'optimal pH and HRT for dark fermentation biohydrogen from food waste'
qemb = model.encode([q], normalize_embeddings=True).astype(np.float32)
labels, dists = p.knn_query(qemb, k=5)

print("Top-5:")
for idx, dist in zip(labels[0], dists[0]):
    m = meta[int(idx)]
    print(m["pdf_file"], m["chunk_id"], "dist=", float(dist))
