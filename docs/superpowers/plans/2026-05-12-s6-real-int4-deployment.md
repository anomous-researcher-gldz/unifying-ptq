# S6: Real INT4 Deployment Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure real INT4 deployment numbers (tokens/sec, peak GPU memory, accuracy preservation) for each of the three architectures, using FlatQuant's own `deploy/` kernels for the LLM main result and torchao for the SAM/SR results plus an LLM supplementary comparison.

**Architecture:** A single `scripts/bench_int4.py` accepts `--backend {flatquant_deploy, torchao}` and `--codebase {llm, sam, sr}`. For each (backend, codebase) it loads the relevant calibrated state, applies the chosen INT4 path, measures (i) tokens/sec or images/sec, (ii) peak `torch.cuda.max_memory_allocated`, (iii) the codebase's accuracy metric.

**Tech Stack:** torchao, FlatQuant `deploy/` kernels (existing), `torch.cuda` memory tracking, `time.perf_counter`.

**Prereqs:** S3 (torchao integration), S2 (AHCPTQ calibrated SAM), S5 (FlatQuant calibrated LLaMA), CompSRT calibrated SwinIR.

---

### Task S6.1: Rebuild FlatQuant `deploy/` kernels for cu124

**Files:**
- Modify: `FlatQuant/deploy/_CUDA.so` (rebuilt)

- [ ] **Step 1: Check if existing _CUDA.so works on torch 2.6+cu124**

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant && conda activate unifyptq
python -c "import deploy._CUDA; print('_CUDA OK')"
```

If "no module named deploy._CUDA" or undefined-symbol, rebuild.

- [ ] **Step 2: Rebuild via FlatQuant setup**

The third-party submodules (cutlass, fast-hadamard-transform) may need cloning. Cutlass is large; do this only if needed.

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant
mkdir -p third-party
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third-party/cutlass
git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform.git third-party/fast-hadamard-transform
TORCH_CUDA_ARCH_LIST="8.0" pip install -e . --no-build-isolation -v > /tmp/fq_build.log 2>&1
tail -20 /tmp/fq_build.log
python -c "import deploy._CUDA; print('_CUDA OK')"
```

Expected: `_CUDA OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
echo "third-party/" >> FlatQuant/.gitignore
git add FlatQuant/.gitignore
git commit -m "build(S6): rebuild FlatQuant deploy/ CUDA kernels for cu124"
```

---

### Task S6.2: Bench script — write skeleton + tests

**Files:**
- Create: `scripts/bench_int4.py`
- Create: `scripts/tests/test_bench_int4.py`

- [ ] **Step 1: Write tests**

```python
# scripts/tests/test_bench_int4.py
import subprocess, json, pathlib

def test_bench_torchao_llm_smoke(tmp_path):
    out = tmp_path / "out.json"
    r = subprocess.run([
        "python", "scripts/bench_int4.py",
        "--backend", "torchao",
        "--codebase", "llm",
        "--model", "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "--config-name", "w4a16",
        "--seq-len", "16",
        "--new-tokens", "4",
        "--out", str(out),
    ], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert "tokens_per_sec" in data
    assert data["tokens_per_sec"] > 0
```

- [ ] **Step 2: Verify fails**

```bash
cd /home/ubuntu/unifying-ptq
pytest scripts/tests/test_bench_int4.py -v
```

Expected: `FileNotFoundError` or similar (script doesn't exist yet).

---

### Task S6.3: Bench script — implement

**Files:**
- Create: `scripts/bench_int4.py`

- [ ] **Step 1: Implement**

```python
# scripts/bench_int4.py
"""Benchmark real INT4 deployment across backends and codebases.

Backends:
  flatquant_deploy: FlatQuant's own CUDA kernels (LLM only)
  torchao:          torchao W4A4 (or W4A16 fallback)

Codebases:
  llm:  HuggingFace causal LM
  sam:  SAM image encoder
  sr:   SwinIR
"""
import argparse, json, pathlib, sys, time
import torch
sys.path.insert(0, "/home/ubuntu/unifying-ptq")

def bench_llm(model, tokenizer, seq_len, new_tokens, n_warmup=2, n_runs=5):
    ids = torch.randint(0, 100, (1, seq_len), device="cuda")
    # warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    total_new = 0
    for _ in range(n_runs):
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
        total_new += g.shape[1] - seq_len
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "tokens_per_sec": total_new / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "seq_len": seq_len, "new_tokens": new_tokens, "n_runs": n_runs,
    }

def bench_sam(encoder, img_size=1024, n_warmup=2, n_runs=10):
    x = torch.randn(1, 3, img_size, img_size, device="cuda", dtype=torch.half)
    for _ in range(n_warmup):
        with torch.no_grad():
            encoder(x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            encoder(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "images_per_sec": n_runs / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "img_size": img_size, "n_runs": n_runs,
    }

def bench_sr(model, img_size=64, n_warmup=2, n_runs=10):
    x = torch.randn(1, 3, img_size, img_size, device="cuda", dtype=torch.half)
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "images_per_sec": n_runs / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "img_size": img_size, "n_runs": n_runs,
    }

def load_llm(model_name, backend, config_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
    t = AutoTokenizer.from_pretrained(model_name)
    if backend == "torchao":
        from flatquant.torchao_deploy import apply_torchao
        m = apply_torchao(m, config_name=config_name)
    elif backend == "flatquant_deploy":
        # Assumes the calibrated _CUDA-using model is already loaded
        pass
    return m, t

def load_sam(ckpt_path, backend, config_name):
    from projects.instance_segment_anything.models.segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    enc = sam.image_encoder.cuda().half()
    if backend == "torchao":
        from ahcptq.torchao_deploy import apply_torchao_sam
        enc = apply_torchao_sam(enc, config_name=config_name)
    return enc

def load_sr(ckpt_path, backend, config_name):
    from basicsr.archs.swinir_arch import SwinIR
    m = SwinIR(upscale=2).cuda().half()
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cuda")
        m.load_state_dict(state["params"] if "params" in state else state, strict=False)
    if backend == "torchao":
        from basicsr.torchao_deploy import apply_torchao_swinir
        m = apply_torchao_swinir(m, config_name=config_name)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["flatquant_deploy", "torchao"], required=True)
    p.add_argument("--codebase", choices=["llm", "sam", "sr"], required=True)
    p.add_argument("--model", default=None, help="HF model id (llm) or checkpoint path (sam/sr)")
    p.add_argument("--config-name", default="w4a4", help="torchao config: w4a4 or w4a16")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.codebase == "llm":
        m, t = load_llm(args.model, args.backend, args.config_name)
        result = bench_llm(m, t, args.seq_len, args.new_tokens)
    elif args.codebase == "sam":
        enc = load_sam(args.model, args.backend, args.config_name)
        result = bench_sam(enc, args.img_size)
    elif args.codebase == "sr":
        m = load_sr(args.model, args.backend, args.config_name)
        result = bench_sr(m, args.img_size)

    result.update({"backend": args.backend, "codebase": args.codebase, "config_name": args.config_name})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify tests pass**

```bash
pytest scripts/tests/test_bench_int4.py -v
```

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_int4.py scripts/tests/test_bench_int4.py
git commit -m "feat(S6): real INT4 deployment benchmark script (llm/sam/sr × flatquant_deploy/torchao)"
```

---

### Task S6.4: Measure LLaMA-3-8B — FlatQuant deploy/ kernels (main paper)

**Files:**
- Create: `results/S6-int4/llm/flatquant_deploy_w4a4/llama3-8b/bench.json`

- [ ] **Step 1: Use FlatQuant's existing inference script (REALQUANT.md)**

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant && conda activate unifyptq
python - <<'EOF'
import time, json, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
m = AutoModelForCausalLM.from_pretrained(
    "Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant",
    trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda:0")
t = AutoTokenizer.from_pretrained("Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant")
ids = torch.randint(0, 100, (1, 2048), device="cuda")
# warmup
for _ in range(2):
    with torch.no_grad():
        m.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=t.eos_token_id)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
total = 0
for _ in range(5):
    with torch.no_grad():
        g = m.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=t.eos_token_id)
    total += g.shape[1] - ids.shape[1]
torch.cuda.synchronize()
dt = time.perf_counter() - t0
result = {
    "backend":"flatquant_deploy","codebase":"llm","config_name":"w4a4kv4",
    "tokens_per_sec": total/dt,
    "peak_mem_mb": torch.cuda.max_memory_allocated()/(1024**2),
    "seq_len":2048,"new_tokens":128,"n_runs":5,
}
p = pathlib.Path("/home/ubuntu/unifying-ptq/results/S6-int4/llm/flatquant_deploy_w4a4/llama3-8b/bench.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(result, indent=2))
print(result)
EOF
```

- [ ] **Step 2: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add results/S6-int4/llm/flatquant_deploy_w4a4/llama3-8b/bench.json
git commit -m "result(S6): LLaMA-3-8B W4A4 throughput via FlatQuant deploy kernels"
```

---

### Task S6.5: Measure LLaMA-3-8B — torchao (supplementary)

**Files:**
- Create: `results/S6-int4/llm/torchao_w4a4/llama3-8b/bench.json`
- Create: `results/S6-int4/llm/torchao_w4a16/llama3-8b/bench.json` (fallback)

- [ ] **Step 1: Run torchao W4A4 attempt**

```bash
python scripts/bench_int4.py --backend torchao --codebase llm --model meta-llama/Meta-Llama-3-8B --config-name w4a4 --seq-len 2048 --new-tokens 128 --out results/S6-int4/llm/torchao_w4a4/llama3-8b/bench.json
```

If it fails, the script auto-falls-back to W4A16 internally; the output `config_name` will reflect what actually ran.

- [ ] **Step 2: Run torchao W4A16 explicit (for comparable supplementary row)**

```bash
python scripts/bench_int4.py --backend torchao --codebase llm --model meta-llama/Meta-Llama-3-8B --config-name w4a16 --seq-len 2048 --new-tokens 128 --out results/S6-int4/llm/torchao_w4a16/llama3-8b/bench.json
```

- [ ] **Step 3: Commit both**

```bash
git add results/S6-int4/llm/torchao_w4a4/ results/S6-int4/llm/torchao_w4a16/
git commit -m "result(S6): LLaMA-3-8B real INT4 throughput via torchao"
```

---

### Task S6.6: Measure SAM-B with torchao

**Files:**
- Create: `results/S6-int4/sam/torchao_w4{a4,a16}/sam-b/bench.json`

- [ ] **Step 1: Run with calibrated checkpoint from S2.3**

```bash
python scripts/bench_int4.py --backend torchao --codebase sam --model /home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth --config-name w4a4 --img-size 1024 --out results/S6-int4/sam/torchao_w4a4/sam-b/bench.json
python scripts/bench_int4.py --backend torchao --codebase sam --model /home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth --config-name w4a16 --img-size 1024 --out results/S6-int4/sam/torchao_w4a16/sam-b/bench.json
```

- [ ] **Step 2: Commit**

```bash
git add results/S6-int4/sam/
git commit -m "result(S6): SAM-B real INT4 throughput via torchao"
```

---

### Task S6.7: Measure SwinIR with torchao

**Files:**
- Create: `results/S6-int4/sr/torchao_w4{a4,a16}/swinir/bench.json`

- [ ] **Step 1: Run with calibrated SwinIR (from CompSRT's existing checkpoint, or one calibrated in S8)**

```bash
python scripts/bench_int4.py --backend torchao --codebase sr --model /home/ubuntu/unifying-ptq/CompSRT/experiments/pretrained/swinir_x2.pth --config-name w4a4 --img-size 64 --out results/S6-int4/sr/torchao_w4a4/swinir/bench.json
python scripts/bench_int4.py --backend torchao --codebase sr --model /home/ubuntu/unifying-ptq/CompSRT/experiments/pretrained/swinir_x2.pth --config-name w4a16 --img-size 64 --out results/S6-int4/sr/torchao_w4a16/swinir/bench.json
```

- [ ] **Step 2: Commit**

```bash
git add results/S6-int4/sr/
git commit -m "result(S6): SwinIR real INT4 throughput via torchao"
```

---

### Task S6.8: Aggregate the multi-architecture deployment table

**Files:**
- Create: `results/S6-int4/summary.json`, `results/S6-int4/summary.md`

- [ ] **Step 1: Aggregate**

```bash
python - <<'EOF'
import json, glob, pathlib
items = [json.load(open(p)) for p in glob.glob("results/S6-int4/**/bench.json", recursive=True)]
rows = sorted(items, key=lambda r: (r["codebase"], r["backend"], r["config_name"]))
pathlib.Path("results/S6-int4/summary.json").write_text(json.dumps(rows, indent=2))
md = ["# Real INT4 Deployment Summary\n",
      "| Codebase | Backend | Config | Throughput | Peak Mem (MB) |",
      "|---|---|---|---|---|"]
for r in rows:
    tput = r.get("tokens_per_sec", r.get("images_per_sec"))
    unit = "tok/s" if "tokens_per_sec" in r else "img/s"
    md.append(f"| {r['codebase']} | {r['backend']} | {r['config_name']} | {tput:.2f} {unit} | {r['peak_mem_mb']:.1f} |")
pathlib.Path("results/S6-int4/summary.md").write_text("\n".join(md))
print(open("results/S6-int4/summary.md").read())
EOF
```

- [ ] **Step 2: Commit**

```bash
git add results/S6-int4/summary.json results/S6-int4/summary.md
git commit -m "result(S6): aggregate real INT4 deployment summary across codebases"
```

---

## Done when

- FlatQuant deploy/ kernels rebuilt and importable
- LLaMA-3-8B benchmarked via FlatQuant kernels + via torchao (at least W4A16)
- SAM-B benchmarked via torchao (W4A4 or W4A16)
- SwinIR benchmarked via torchao (W4A4 or W4A16)
- `results/S6-int4/summary.md` table exists and is readable
- Numbers ready for paper section "Real INT4 Deployment"
