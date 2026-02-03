import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60] or "query"

def load_chunks_dict(chunks_path: Path):
    d = {}
    with chunks_path.open("r", encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            d[rec["chunk_id"]] = {"text": rec["text"], "pdf_file": rec["pdf_file"], "doc_id": rec.get("doc_id","")}
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--index_dir", type=str, default="data/index_bruteforce_q1")
    ap.add_argument("--chunks", type=str, default="data/processed/chunks_q1.jsonl")
    ap.add_argument("--docmeta", type=str, default="data/processed/docmeta_q1.json")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    emb_path = index_dir / "emb.npy"
    meta_path = index_dir / "meta.json"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("emb.npy/meta.json not found. Run scripts/03_embed_and_query_bruteforce.py first.")

    # embeddings (mmap to save memory)
    emb = np.load(emb_path, mmap_mode="r")  # (N, D) float32
    with meta_path.open("r", encoding="utf-8") as r:
        meta_blob = json.load(r)
    meta_list = meta_blob["meta"]
    model_name = meta_blob.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    # chunk text dict
    chunks_dict = load_chunks_dict(Path(args.chunks))

    # docmeta
    docmeta = {}
    dp = Path(args.docmeta)
    if dp.exists():
        with dp.open("r", encoding="utf-8") as r:
            docmeta = json.load(r)

    # encode query
    model = SentenceTransformer(model_name)
    q = args.query
    qemb = model.encode([q], normalize_embeddings=True).astype(np.float32)[0]

    sims = np.asarray(emb @ qemb)  # cosine similarity because normalized
    k = min(args.k, sims.shape[0])
    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    # build pack
    results = []
    for rank, i in enumerate(idx, 1):
        m = meta_list[int(i)]
        chunk_id = m["chunk_id"]
        pdf_file = m["pdf_file"]
        ch = chunks_dict.get(chunk_id, {"text": "", "pdf_file": pdf_file})
        dm = docmeta.get(pdf_file, {"doi": "", "title": "", "year": None, "venue": "", "authors": ""})

        results.append({
            "rank": rank,
            "sim": float(sims[int(i)]),
            "chunk_id": chunk_id,
            "pdf_file": pdf_file,
            "doi": dm.get("doi",""),
            "title": dm.get("title",""),
            "year": dm.get("year", None),
            "venue": dm.get("venue",""),
            "authors": dm.get("authors",""),
            "text": ch.get("text","")
        })

    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{slug(q)}"
    out_json = runs / f"{base}.json"
    out_md = runs / f"{base}.md"

    with out_json.open("w", encoding="utf-8") as w:
        json.dump({"query": q, "k": k, "results": results}, w, ensure_ascii=False, indent=2)

    # markdown for manual paste into LLM (or later API)
    with out_md.open("w", encoding="utf-8") as w:
        w.write(f"# RAG Evidence Pack\n\n")
        w.write(f"**Query:** {q}\n\n")
        w.write(f"**Top-{k} Results:**\n\n")
        for r in results:
            cite = []
            if r["authors"]: cite.append(r["authors"])
            if r["title"]: cite.append(r["title"])
            if r["venue"] or r["year"]:
                vy = " ".join([x for x in [r["venue"], str(r["year"]) if r["year"] else ""] if x])
                if vy: cite.append(vy)
            if r["doi"]: cite.append(f"DOI: {r['doi']}")
            cite_line = " | ".join(cite) if cite else "(no docmeta matched)"

            w.write(f"## [{r['rank']}] sim={r['sim']:.4f}  {r['pdf_file']}  {r['chunk_id']}\n\n")
            w.write(f"**Citation:** {cite_line}\n\n")
            w.write("```text\n")
            w.write((r["text"] or "").strip() + "\n")
            w.write("```\n\n")

    print("Wrote:", out_json)
    print("Wrote:", out_md)

if __name__ == "__main__":
    main()
