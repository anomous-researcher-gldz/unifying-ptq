"""Item 3 — Real INT4 GEMM kernel benchmark.

Uses torchao's int4_dynamic_activation_int4_weight (W4A4) — the same kernel
that's deployed in vLLM / SGLang / production inference servers.

Compares the relative cost of each primitive ON TOP of the real INT4 kernel:
  - none           : just the W4A4 INT4 GEMM (production baseline)
  - hadamard       : extra fp16 (d x d) matmul applied BEFORE the INT4 GEMM
  - learned_R      : extra fp16 (d x d) matmul applied BEFORE the INT4 GEMM
  - dbaf           : in-place fp16 fold applied BEFORE the INT4 GEMM
  - pcsa_tf        : in-place fp16 scalar mul applied BEFORE the INT4 GEMM
  - dbaf_pcsa_tf   : both in-place fp16 ops applied BEFORE the INT4 GEMM

Insight: rotation forces an ADDITIONAL dense fp16 matmul kernel launch on
top of the INT4 GEMM.  DBAF/PCSA-tf are element-wise ops that the JIT can
fuse into the dequant pre-amble (or simply launch as a single cheap kernel).

For the paper: this is the "Strong" credibility tier — the kernel used in
real deployment, with our primitives as a per-block input pre-processor.

Output:
  scripts/_out/int4_gemm_latency.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, time
import torch
import torch.nn as nn


def _hadamard(d: int, device, dtype):
    assert (d & (d - 1)) == 0
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(d)


def _random_R(d: int, device, dtype):
    G = torch.randn(d, d, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q.to(dtype)


class _BenchBlock(nn.Module):
    """One Linear(d, d).  The torchao quantize_ pass will replace its weight
    with an INT4 tensor wrapper that calls the fast kernel."""
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.lin.weight, std=0.02)

    def forward(self, x):
        return self.lin(x)


class _Stack(nn.Module):
    def __init__(self, d: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([_BenchBlock(d) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def _apply_primitive(x: torch.Tensor, prim: str, st: dict) -> torch.Tensor:
    if prim == "none":
        return x
    if prim == "hadamard":
        return x @ st["H"]
    if prim == "learned_R":
        return x @ st["R"]
    if prim == "dbaf":
        T = st.get("dbaf_T")
        if T is None:
            T = 3.0 * x.std().item()
            st["dbaf_T"] = T
        a = st["dbaf_alpha"]
        sgn = torch.sign(x); mask = x.abs() > T
        return torch.where(mask, sgn * T + a * (x - sgn * T), x)
    if prim == "pcsa_tf":
        return x * st["scales"][0]
    if prim == "dbaf_pcsa_tf":
        T = st.get("dbaf_T")
        if T is None:
            T = 3.0 * x.std().item()
            st["dbaf_T"] = T
        a = st["dbaf_alpha"]
        sgn = torch.sign(x); mask = x.abs() > T
        x = torch.where(mask, sgn * T + a * (x - sgn * T), x)
        return x * st["scales"][0]
    raise ValueError(prim)


def _build_state(prim: str, d: int, device, dtype, K: int = 4) -> dict:
    st = {"dbaf_alpha": 0.95}
    if prim == "hadamard":
        st["H"] = _hadamard(d, device, dtype)
    if prim == "learned_R":
        st["R"] = _random_R(d, device, dtype)
    if prim in ("pcsa_tf", "dbaf_pcsa_tf"):
        st["anchors"] = torch.randn(K, d, device=device, dtype=dtype)
        st["scales"] = torch.rand(K, device=device, dtype=dtype) + 0.5
        st["desc"] = torch.randn(d, device=device, dtype=dtype)
    return st


def _make_hook(prim: str, state: dict):
    def hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        return (_apply_primitive(x, prim, state),) + tuple(inputs[1:])
    return hook


def bench_one(prim: str, model_q: nn.Module, d: int, n_blocks: int,
              seq_len: int, n_warmup: int, n_iters: int,
              device, dtype) -> dict:
    state = _build_state(prim, d, device, dtype)

    # Attach hook on each Linear's input
    handles = []
    if prim != "none":
        for blk in model_q.blocks:
            handles.append(blk.lin.register_forward_pre_hook(_make_hook(prim, state)))

    x0 = torch.randn(1, seq_len, d, device=device, dtype=dtype)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model_q(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model_q(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    for h in handles:
        h.remove()
    return {"primitive": prim, "ms_per_forward": (t1 - t0) * 1000.0 / n_iters,
            "n_iters": n_iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d",        type=int, default=4096)
    ap.add_argument("--n_blocks", type=int, default=32)
    ap.add_argument("--seq_len",  type=int, default=4096)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_iters",  type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, default="scripts/_out/int4_gemm_latency.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "INT4 GEMM kernel needs CUDA"
    device = torch.device("cuda")
    dtype = torch.bfloat16  # torchao W4A4 path prefers bf16

    # Build the stack and quantize via torchao's W4A4 API
    from torchao.quantization import quantize_, int4_dynamic_activation_int4_weight
    stack = _Stack(args.d, args.n_blocks).to(device=device, dtype=dtype)
    # baseline (bf16, no quant)
    primitives = ["none", "hadamard", "learned_R", "dbaf", "pcsa_tf", "dbaf_pcsa_tf"]

    # Run bf16 baseline first (un-quantized)
    print(f"\n=== bf16 (no quant) baseline d={args.d}, n_blocks={args.n_blocks}, seq_len={args.seq_len} ===")
    bf16_rows = [bench_one(p, stack, args.d, args.n_blocks, args.seq_len,
                            args.n_warmup, args.n_iters, device, dtype) | {"backend": "bf16"}
                  for p in primitives]
    for r in bf16_rows:
        print(f"  bf16     {r['primitive']:20s}  {r['ms_per_forward']:9.2f} ms")

    # Apply torchao W4A4 INT4 GEMM in-place
    print("\n=== Applying torchao int4_dynamic_activation_int4_weight ===")
    try:
        quantize_(stack, int4_dynamic_activation_int4_weight())
        backend = "int4"
    except Exception as exc:
        print(f"  WARNING: int4_dynamic_activation_int4_weight FAILED: {exc}")
        print("  falling back to int4_weight_only (W4A16, weight-only kernel)")
        from torchao.quantization import int4_weight_only
        quantize_(stack, int4_weight_only())
        backend = "int4_woq"

    print(f"\n=== {backend} (torchao) d={args.d}, n_blocks={args.n_blocks}, seq_len={args.seq_len} ===")
    int4_rows = [bench_one(p, stack, args.d, args.n_blocks, args.seq_len,
                            args.n_warmup, args.n_iters, device, dtype) | {"backend": backend}
                  for p in primitives]
    for r in int4_rows:
        print(f"  {backend:8s} {r['primitive']:20s}  {r['ms_per_forward']:9.2f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"primitives": primitives, "dtype": str(dtype)},
        "bf16_results": bf16_rows,
        "int4_results": int4_rows,
    }, indent=2, default=str))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
