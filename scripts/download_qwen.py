import os
from huggingface_hub import snapshot_download

repo = "Qwen/Qwen2.5-7B-Instruct"
print("[dl] endpoint =", os.environ.get("HF_ENDPOINT"), flush=True)
print("[dl] hf_home   =", os.environ.get("HF_HOME"), flush=True)

path = snapshot_download(
    repo_id=repo,
    resume_download=True,
    max_workers=1,          # 关键：单线程最稳
)
print("[dl] done ->", path, flush=True)
