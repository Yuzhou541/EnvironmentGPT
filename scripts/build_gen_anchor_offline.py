import json, random
from pathlib import Path

random.seed(7)
out = Path("data/train/gen_anchor.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

topics = [
  "mathematics", "physics", "chemistry", "biology", "computer science",
  "writing", "history", "geography", "economics", "logic", "programming",
  "data analysis", "statistics", "machine learning", "ethics", "education"
]

templates = [
  "Explain the core concept of {topic} in simple terms, with one example.",
  "Compare two approaches in {topic} and discuss trade-offs.",
  "Given the following constraints, propose a plan and justify it: {topic}.",
  "Provide a step-by-step solution strategy for a typical problem in {topic}.",
  "Summarize the key points and common pitfalls in {topic}.",
  "Rewrite the following idea more clearly and professionally: {topic}.",
  "Generate three quiz questions (with short answers) about {topic}.",
  "Provide an outline for a short tutorial about {topic} for beginners.",
  "Give an intuitive explanation and then a formal explanation of {topic}."
]

# 额外一些通用指令（更贴近真实用户分布）
generic_prompts = [
  "Draft a concise email requesting a meeting and include a polite tone.",
  "Help me debug a Python error: 'IndexError: list index out of range'. Explain typical causes.",
  "I have 3 hours to study today. Create a focused study plan.",
  "Explain how gradient descent works and why learning rate matters.",
  "Describe how to design an ablation study for a machine learning paper.",
  "Explain the difference between precision, recall, and F1 score with an example.",
  "Provide a checklist for writing a strong methods section in an NLP paper."
]

N = 80000
cnt = 0
with out.open("w", encoding="utf-8") as w:
    # 先写入固定的 generic
    for gp in generic_prompts:
        w.write(json.dumps({"type":"gen","prompt":gp,"response":""}, ensure_ascii=False) + "\n")
        cnt += 1

    # 再用模板组合扩充到 N
    while cnt < N:
        t = random.choice(templates)
        topic = random.choice(topics)
        prompt = t.format(topic=topic)
        # 增加一点随机性
        if random.random() < 0.35:
            prompt += " Provide a short example."
        if random.random() < 0.20:
            prompt += " Keep it under 150 words."
        w.write(json.dumps({"type":"gen","prompt":prompt,"response":""}, ensure_ascii=False) + "\n")
        cnt += 1

print(f"[gen_anchor_offline] samples={cnt} out={out}")
