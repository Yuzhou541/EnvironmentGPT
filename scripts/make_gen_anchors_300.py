import argparse, json, random, os

TOPICS = [
  "basic algebra", "probability", "linear regression", "gradient descent", "matrix multiplication",
  "data structures", "time complexity", "Python debugging", "Linux shell", "Git workflow",
  "HTTP vs HTTPS", "TCP vs UDP", "database indexing", "distributed systems", "unit testing",
  "writing clarity", "email etiquette", "summarization", "argument evaluation", "critical thinking",
  "climate basics", "water cycle", "carbon cycle", "ecosystems", "renewable energy",
  "statistics concepts", "hypothesis testing", "confidence intervals", "bias-variance tradeoff", "overfitting",
  "neural networks", "transformers", "attention mechanism", "prompting", "model evaluation",
  "math proof style", "induction", "contradiction", "limits", "derivatives",
  "physics basics", "electricity", "magnetism", "optics", "thermodynamics",
  "chemistry basics", "acids and bases", "equilibrium", "reaction rates", "stoichiometry",
  "project planning", "risk management", "stakeholder update", "timeline estimation", "trade-off analysis"
]

TEMPLATES = [
  "Explain {topic} in simple terms with one concrete example.",
  "Give a concise step-by-step method for {topic}.",
  "Compare two common approaches related to {topic} and state when each is preferable.",
  "List 5 common mistakes in {topic} and how to avoid them.",
  "Provide a short checklist for doing {topic} well.",
  "Summarize {topic} in 5 bullet points, each under 12 words.",
  "Write a brief Q&A (3 questions) to test understanding of {topic}.",
  "Give a real-world scenario where {topic} matters, and explain why.",
  "Provide a mini-lesson on {topic} suitable for a high school student.",
  "Explain {topic} using an analogy, then restate without the analogy."
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    random.seed(args.seed)

    prompts = []
    # generate deterministically by cycling topics + random templates
    i = 0
    while len(prompts) < args.n:
        topic = TOPICS[i % len(TOPICS)]
        tpl = random.choice(TEMPLATES)
        p = tpl.format(topic=topic)
        prompts.append(p)
        i += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for j, p in enumerate(prompts):
            obj = {"id": f"gen_anchor_{j:03d}", "prompt": p}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(prompts)} anchors -> {args.out}")

if __name__ == "__main__":
    main()
