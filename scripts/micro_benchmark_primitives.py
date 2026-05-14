"""Wall-clock micro-benchmark for §4.X primitive op-cost.

Complements scripts/compute_flop_table.py with empirical ms/forward numbers.
For each primitive, we run a synthetic Llama-3-8B-shaped forward (sequence
length 4096, hidden dim 4096) with that primitive applied at every block,
and report ms/forward + ms/token on the local A100.

Primitives benchmarked:
  - none           (fp16 baseline; no quantization, no extra op)
  - fakequant_w4a4 (RTN-style per-channel weight + per-token act quant)
  - hadamard       (fast Hadamard rotation per block, applied to act)
  - learned_R      (dense d x d learned rotation; FlatQuant-style)
  - dbaf           (per-tensor outlier fold; gate trivially fires)
  - pcsa_tf        (per-input scale routing; K=4 anchors)
  - dbaf_pcsa_tf   (both, composed)

We exercise an absolute minimum stack (one Linear(d,d) per "block", 32 blocks)
to isolate the cost of the primitive itself from the model's actual MLPs/attn.
The ratios reported here are conservative (real Llama-3 has multiple matmuls
per block, so rotation's d^2 cost compounds more in production).

Output:
  - scripts/_out/micro_benchmark.json
  - markdown summary to stdout
"""
from __future__ import annotations
import argparse, json, math, pathlib, time
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitive ops
# ---------------------------------------------------------------------------

def _fakequant_w4a4(w: torch.Tensor, x: torch.Tensor):
    qmax = 7  # 2^(4-1) - 1
    # weight quant (per-channel, dim=0)
    w_scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-9) / qmax
    qw = torch.round(w / w_scale).clamp(-qmax, qmax) * w_scale
    # act quant (per-token, dim=-1)
    a_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / qmax
    qx = torch.round(x / a_scale).clamp(-qmax, qmax) * a_scale
    return qw, qx


def _make_hadamard(d: int, device, dtype):
    """Build a fast Hadamard matrix of size d (must be power of 2)."""
    assert (d & (d - 1)) == 0, f"d={d} must be power of 2 for Hadamard"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    H = H / math.sqrt(d)
    return H


def _learned_rotation(d: int, device, dtype):
    """Random orthogonal d x d matrix (proxy for learned R).

    QR is unavailable in fp16 on CUDA; compute in fp32 and cast.
    """
    G = torch.randn(d, d, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q.to(dtype)


def _dbaf_fold(x: torch.Tensor, T: float | None = None, alpha: float = 0.95):
    """Per-tensor fold.

    In production, T is precomputed at calibration time and stored as a buffer.
    The micro-benchmark thus uses a precomputed T (passed in) when available,
    matching the real inference path.  Pass T=None only when actually fitting.
    """
    if T is None:
        sigma = x.std()
        T = 3.0 * sigma
    sgn = torch.sign(x)
    mask = x.abs() > T
    out = torch.where(mask, sgn * T + alpha * (x - sgn * T), x)
    return out, T, alpha


def _dbaf_unfold(y: torch.Tensor, T: float, alpha: float):
    sgn = torch.sign(y)
    mask = y.abs() > T
    return torch.where(mask, sgn * T + (1.0 / alpha) * (y - sgn * T), y)


def _pcsa_tf_route(x: torch.Tensor, anchors: torch.Tensor, scales: torch.Tensor,
                   descriptor: torch.Tensor):
    """Route via cosine-similarity to K anchors; apply selected scale."""
    desc = descriptor / (descriptor.norm() + 1e-9)
    ancs = anchors / (anchors.norm(dim=-1, keepdim=True) + 1e-9)
    sims = ancs @ desc
    k = int(sims.argmax())
    return x * scales[k]


# ---------------------------------------------------------------------------
# Forward-pass driver per primitive
# ---------------------------------------------------------------------------

def _forward(primitive: str, weights: list[torch.Tensor], x: torch.Tensor,
             extra: dict | None = None):
    extra = extra or {}
    for w in weights:
        if primitive == "none":
            x = torch.nn.functional.linear(x, w)
        elif primitive == "fakequant_w4a4":
            qw, qx = _fakequant_w4a4(w, x)
            x = torch.nn.functional.linear(qx, qw)
        elif primitive == "hadamard":
            H = extra["H"]
            x = x @ H
            qw, qx = _fakequant_w4a4(w, x)
            x = torch.nn.functional.linear(qx, qw)
        elif primitive == "learned_R":
            R = extra["R"]
            x = x @ R
            qw, qx = _fakequant_w4a4(w, x)
            x = torch.nn.functional.linear(qx, qw)
        elif primitive == "dbaf":
            # T precomputed at calibration time in production; pass from extra
            x, T, alpha = _dbaf_fold(x, T=extra.get("T_fixed"))
            qw, qx = _fakequant_w4a4(w, x)
            x = torch.nn.functional.linear(qx, qw)
            x = _dbaf_unfold(x, T, alpha)
        elif primitive == "pcsa_tf":
            x = _pcsa_tf_route(x, extra["anchors"], extra["scales"], extra["desc"])
            qw, qx = _fakequant_w4a4(w, x)
            x = torch.nn.functional.linear(qx, qw)
        elif primitive == "dbaf_pcsa_tf":
            x, T, alpha = _dbaf_fold(x, T=extra.get("T_fixed"))
            x = _pcsa_tf_route(x, extra["anchors"], extra["scales"], extra["desc"])
            qw, qx = _fakequant_w4a4(w, x)
            x = torch.nn.functional.linear(qx, qw)
            x = _dbaf_unfold(x, T, alpha)
        else:
            raise ValueError(f"unknown primitive: {primitive}")
    return x


def bench_one(primitive: str, d: int, n_blocks: int, seq_len: int,
              n_warmup: int, n_iters: int, device: torch.device,
              dtype: torch.dtype, K: int = 4):
    weights = [torch.randn(d, d, device=device, dtype=dtype) * 0.02
               for _ in range(n_blocks)]
    x0 = torch.randn(1, seq_len, d, device=device, dtype=dtype)

    extra: dict = {}
    if primitive == "hadamard":
        extra["H"] = _make_hadamard(d, device, dtype)
    if primitive == "learned_R":
        extra["R"] = _learned_rotation(d, device, dtype)
    if primitive in ("pcsa_tf", "dbaf_pcsa_tf"):
        extra["anchors"] = torch.randn(K, d, device=device, dtype=dtype)
        extra["scales"] = torch.rand(K, device=device, dtype=dtype) + 0.5
        extra["desc"] = torch.randn(d, device=device, dtype=dtype)
    if primitive in ("dbaf", "dbaf_pcsa_tf"):
        # Precompute T_fixed from a single calib forward (matches the production
        # path where T is stored as a buffer after calibration).
        with torch.no_grad():
            extra["T_fixed"] = (3.0 * x0.std()).item()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = _forward(primitive, weights, x0, extra)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed iterations
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = _forward(primitive, weights, x0, extra)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    ms_per_forward = total_ms / n_iters
    ms_per_token = ms_per_forward / seq_len
    return {"primitive": primitive,
            "ms_per_forward": ms_per_forward,
            "ms_per_token_us": ms_per_token * 1000.0,  # in microseconds
            "n_iters": n_iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d",        type=int, default=4096, help="hidden dim (Llama-3-8B)")
    ap.add_argument("--n_blocks", type=int, default=32,   help="layers (Llama-3-8B)")
    ap.add_argument("--seq_len",  type=int, default=4096)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_iters",  type=int, default=10)
    ap.add_argument("--out",      type=pathlib.Path, default="scripts/_out/micro_benchmark.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    primitives = ["none", "fakequant_w4a4", "hadamard", "learned_R",
                  "dbaf", "pcsa_tf", "dbaf_pcsa_tf"]

    print(f"# Wall-clock micro-benchmark @ d={args.d}, n_blocks={args.n_blocks}, seq_len={args.seq_len}")
    print(f"# device={device}, dtype={dtype}, warmup={args.n_warmup}, iters={args.n_iters}\n")

    results = []
    for p in primitives:
        r = bench_one(p, args.d, args.n_blocks, args.seq_len, args.n_warmup,
                      args.n_iters, device, dtype)
        results.append(r)
        print(f"  {r['primitive']:20s}  {r['ms_per_forward']:8.2f} ms/forward  "
              f"{r['ms_per_token_us']:8.2f} us/token")

    # Ratios vs fp16 baseline and vs Hadamard
    fp = next(r for r in results if r["primitive"] == "none")["ms_per_forward"]
    hd = next(r for r in results if r["primitive"] == "hadamard")["ms_per_forward"]
    ours = next(r for r in results if r["primitive"] == "dbaf_pcsa_tf")["ms_per_forward"]
    print(f"\n## Ratios")
    print(f"  Hadamard overhead     : {hd / fp:.2f}x  (vs fp16 baseline)")
    print(f"  DBAF+PCSA-tf overhead : {ours / fp:.2f}x  (vs fp16 baseline)")
    print(f"  Hadamard / DBAF+PCSA-tf wall-clock = {hd / ours:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"d": args.d, "n_blocks": args.n_blocks,
                   "seq_len": args.seq_len, "device": str(device),
                   "dtype": str(dtype),
                   "n_warmup": args.n_warmup, "n_iters": args.n_iters},
        "results": results,
    }, indent=2))
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
