import json
import random
from pathlib import Path

facts_path = Path("data/processed/facts_ph_hrt_q1.jsonl")
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

qa_all = out_dir / "qa_env_q1_all.jsonl"
qa_train = out_dir / "qa_env_q1_train.jsonl"
qa_dev = out_dir / "qa_env_q1_dev.jsonl"
qa_test = out_dir / "qa_env_q1_test.jsonl"

random.seed(42)

def cite_str(f):
    title = (f.get("title") or "").strip()
    year = f.get("year", None)
    doi = (f.get("doi") or "").strip()
    parts = []
    if title: parts.append(title)
    if year: parts.append(str(year))
    if doi: parts.append(f"DOI: {doi}")
    return " | ".join(parts) if parts else f.get("pdf_file","")

Q_PH = [
    "According to {CITE}, what pH (or pH range) is reported as optimal for dark fermentation biohydrogen production?",
    "In {CITE}, what is the reported optimal pH condition for dark fermentation biohydrogen production?"
]
Q_HRT = [
    "According to {CITE}, what hydraulic retention time (HRT) is reported as optimal for dark fermentation biohydrogen production?",
    "In {CITE}, what is the reported optimal HRT condition for dark fermentation biohydrogen production?"
]

def make_qa(f):
    t = f["type"]
    pdf = f["pdf_file"]
    chunk = f["chunk_id"]
    lo, hi = f["value_min"], f["value_max"]
    unit = f.get("unit","")

    CITE = cite_str(f)

    if t.startswith("pH"):
        q = random.choice(Q_PH).format(CITE=CITE)
        if lo == hi:
            a = f"Reported optimal pH: {lo:g}."
        else:
            a = f"Reported optimal pH range: {lo:g}-{hi:g}."
    elif t.startswith("HRT"):
        q = random.choice(Q_HRT).format(CITE=CITE)
        unit_str = "hours" if unit == "h" else "days" if unit == "d" else unit
        if lo == hi:
            a = f"Reported optimal HRT: {lo:g} {unit_str}."
        else:
            a = f"Reported optimal HRT range: {lo:g}-{hi:g} {unit_str}."
    else:
        return None

    cite = {
        "pdf_file": pdf,
        "chunk_id": chunk,
        "doi": f.get("doi",""),
        "title": f.get("title",""),
        "year": f.get("year", None),
    }

    return {
        "question": q,
        "answer": a,
        "citations": [cite],
        "meta": {
            "type": t,
            "value_min": lo,
            "value_max": hi,
            "unit": unit,
            "signal": bool(f.get("signal", False)),
            "cite_str": CITE,
        }
    }

qas, seen = [], set()
with facts_path.open("r", encoding="utf-8") as r:
    for line in r:
        f = json.loads(line)
        qa = make_qa(f)
        if qa is None:
            continue
        key = (qa["meta"]["type"], qa["citations"][0].get("doi",""), qa["meta"]["value_min"], qa["meta"]["value_max"], qa["meta"]["unit"])
        if key in seen:
            continue
        seen.add(key)
        qas.append(qa)

random.shuffle(qas)

def dump(path: Path, items):
    with path.open("w", encoding="utf-8") as w:
        for x in items:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("Wrote:", path, "lines=", len(items))

n = len(qas)
n_train = max(1, int(0.8 * n))
n_dev = max(1, int(0.1 * n)) if n >= 10 else 1
train = qas[:n_train]
dev = qas[n_train:n_train+n_dev]
test = qas[n_train+n_dev:]

dump(qa_all, qas)
dump(qa_train, train)
dump(qa_dev, dev)
dump(qa_test, test)
