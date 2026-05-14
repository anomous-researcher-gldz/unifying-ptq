"""Per-layer DBAF ablation for LLaMA-3-8B Linear weights.

For each Linear layer in the model:
  baseline  = no-DBAF (RTN per-channel on every Linear)
  per_layer = DBAF applied to EXACTLY ONE layer; RTN on all others.

Evaluates WikiText-2 chunk perplexity on a small subset (default 8 chunks × 2048 tokens)
for speed.  Does NOT require a GPU — runs in fp16/cpu mode or on CUDA if available,
but is designed to be written/syntax-checked without running.

Output JSON mirrors the SwinIR + SAM ablation structure:
  {
    "summary": {...},
    "rows": [{"layer", "shape", "frac3", "gate", "metric", "delta"}, ...]
  }

where "metric" is the chunk PPL and "delta" is PPL_layer - PPL_baseline
(negative delta = DBAF helps; positive delta = DBAF hurts).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")

from ahcptq.quantization.fake_quant import is_like_normal_plus_3sigma_outliers
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf

# Default model path — override with --model-path
DEFAULT_MODEL = "/home/ubuntu/unifying-ptq/ckpt/meta-llama/Meta-Llama-3-8B"
DEFAULT_WIKITEXT = "/home/ubuntu/unifying-ptq/data/wikitext2"


# ── quantisation helpers ──────────────────────────────────────────────────
def _rtn_layer(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel RTN quantisation."""
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1)
    return _quantize_tensor_uniform(w_flat, bits, per_channel=True).view_as(w).to(w.dtype)


def _dbaf_layer(w: torch.Tensor, bits: int, alpha: float) -> torch.Tensor:
    """Per-output-channel DBAF quantisation."""
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1)
    return _quantize_per_channel_with_dbaf(w_flat.float(), bits, alpha=alpha).view_as(w).to(w.dtype)


# ── model helpers ─────────────────────────────────────────────────────────
def load_model(model_path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[llama-ablation] loading model from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def list_target_layers(model: nn.Module) -> list[str]:
    """Return names of all Linear layers with >= 64 weights (transformer weight matrices)."""
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 64:
            names.append(name)
    return names


def snapshot_fp_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 64:
            out[name] = mod.weight.data.clone()
    return out


def apply_quant_with_one_dbaf(
    model: nn.Module,
    fp_weights: dict[str, torch.Tensor],
    dbaf_layer: Optional[str],
    bits: int,
    alpha: float,
) -> None:
    """Reload FP weights, quantize: DBAF for dbaf_layer, RTN for all others."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 64:
            w_fp = fp_weights[name]
            w_q = (
                _dbaf_layer(w_fp, bits, alpha)
                if name == dbaf_layer
                else _rtn_layer(w_fp, bits)
            )
            mod.weight.data = w_q.to(mod.weight.dtype).clone()


# ── WikiText-2 perplexity ─────────────────────────────────────────────────
def load_wikitext2_tokens(tokenizer, data_dir: str, seq_len: int, n_chunks: int) -> torch.Tensor:
    """Load WikiText-2 test set, tokenize, return [n_chunks, seq_len] token tensor."""
    test_file = pathlib.Path(data_dir) / "wiki.test.tokens"
    if not test_file.exists():
        # Fallback: try to load via datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(ds["text"])
        except Exception as exc:
            raise FileNotFoundError(
                f"WikiText-2 test file not found at {test_file} and datasets fallback failed: {exc}"
            )
    else:
        text = test_file.read_text(encoding="utf-8")

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokens = enc["input_ids"].squeeze(0)  # [total_len]
    total_needed = n_chunks * seq_len
    if tokens.numel() < total_needed:
        # Repeat to fill (rare for WikiText-2 test with small n_chunks)
        repeats = (total_needed // tokens.numel()) + 1
        tokens = tokens.repeat(repeats)
    tokens = tokens[:total_needed].view(n_chunks, seq_len)
    return tokens


@torch.no_grad()
def compute_chunk_ppl(model: nn.Module, tokens: torch.Tensor) -> float:
    """Compute mean per-token negative log-likelihood over chunks, return PPL."""
    n_chunks, seq_len = tokens.shape
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for i in range(n_chunks):
        chunk = tokens[i].unsqueeze(0).to(device)  # [1, seq_len]
        out = model(chunk, labels=chunk)
        # out.loss is mean NLL over (seq_len - 1) tokens
        nll = out.loss.item() * (seq_len - 1)
        total_nll += nll
        total_tokens += seq_len - 1
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
    return ppl


# ── main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Per-layer DBAF ablation for LLaMA-3-8B (WikiText-2 chunk PPL)"
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--wikitext-dir", default=DEFAULT_WIKITEXT)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--n-chunks", type=int, default=8, help="Number of 2048-token chunks for PPL eval")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-layers", type=int, default=None, help="Cap ablation layers (for testing)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    model, tokenizer = load_model(args.model_path, device=args.device)
    fp_weights = snapshot_fp_weights(model)
    layers = list_target_layers(model)
    if args.max_layers:
        layers = layers[: args.max_layers]

    print(
        f"[llama-ablation] model={pathlib.Path(args.model_path).name} "
        f"bits={args.bits} alpha={args.alpha} "
        f"n_chunks={args.n_chunks} seq_len={args.seq_len} "
        f"target_layers={len(layers)}",
        flush=True,
    )

    tokens = load_wikitext2_tokens(
        tokenizer, args.wikitext_dir, args.seq_len, args.n_chunks
    )
    print(f"[llama-ablation] loaded {tokens.numel()} tokens for PPL eval", flush=True)

    # ── baseline: RTN everywhere, no DBAF ──────────────────────────────────
    apply_quant_with_one_dbaf(model, fp_weights, dbaf_layer=None, bits=args.bits, alpha=args.alpha)
    t0 = time.time()
    ppl_baseline = compute_chunk_ppl(model, tokens)
    print(f"[baseline] no-DBAF PPL = {ppl_baseline:.4f} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for i, name in enumerate(layers):
        fp = fp_weights[name]
        flat = fp.detach().float().reshape(-1)
        mu, sd = flat.mean(), flat.std().clamp_min(1e-8)
        z = (flat - mu) / sd
        frac3 = float((z.abs() > 3.0).float().mean().item())
        gate = bool(is_like_normal_plus_3sigma_outliers(fp)["is_like_c"])

        apply_quant_with_one_dbaf(
            model, fp_weights, dbaf_layer=name, bits=args.bits, alpha=args.alpha
        )
        ppl_l = compute_chunk_ppl(model, tokens)
        delta = ppl_l - ppl_baseline  # negative = DBAF helps

        rows.append(
            {
                "layer": name,
                "shape": list(fp.shape),
                "frac3": frac3,
                "gate": gate,
                "metric": ppl_l,
                "delta": delta,
            }
        )
        if (i + 1) % 10 == 0 or i == len(layers) - 1:
            print(
                f"  [{i+1:3d}/{len(layers)}] {name}: gate={gate} frac3={frac3:.4f} "
                f"PPL={ppl_l:.4f} Δ={delta:+.4f}",
                flush=True,
            )

    gate_pass = [r["delta"] for r in rows if r["gate"]]
    gate_fail = [r["delta"] for r in rows if not r["gate"]]
    summary = {
        "model": pathlib.Path(args.model_path).name,
        "bits": args.bits,
        "alpha": args.alpha,
        "n_chunks": args.n_chunks,
        "seq_len": args.seq_len,
        "baseline_ppl": ppl_baseline,
        "n_layers": len(rows),
        "n_gate_pass": len(gate_pass),
        "n_gate_fail": len(gate_fail),
        "mean_delta_gate_pass": float(np.mean(gate_pass)) if gate_pass else None,
        "mean_delta_gate_fail": float(np.mean(gate_fail)) if gate_fail else None,
        "max_delta": max(r["delta"] for r in rows) if rows else None,
        "min_delta": min(r["delta"] for r in rows) if rows else None,
        "n_positive_delta": sum(1 for r in rows if r["delta"] > 0),
        "n_negative_delta": sum(1 for r in rows if r["delta"] < 0),
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
