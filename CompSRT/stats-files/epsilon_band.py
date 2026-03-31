#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
epsilon_band_padded.py  (pairing-aware)

Goal
----
Test whether the *fraction* of values within [-eps, eps] **increases after the transform**.
Experimental unit = entire tensor (matrix/filter). Elements are not indexwise matchable
post-Hadamard, so we aggregate *per tensor* and then do a paired test across tensors.

Per tensor j:
  p_pre(j)  = mean( |pre|  <= eps )      over finite entries
  p_post(j) = mean( |post| <= eps )      over finite entries
  delta_j   = p_post(j) - p_pre(j)

Across tensors, we test H0: median(delta)=0 vs H1: median(delta)>0 (Wilcoxon signed-rank, one-sided).
We also report an exact sign test (binomial on wins) and Cohen's d_z on the delta vector.

CLI
---
python epsilon_band_padded.py /path/to/dir --eps 1e-3 --by-type --pad-mode ignore_pre_padding

Pairing rule (non-recursive):  *_prehad.pt  ->  *_posthad.pt
Filename convention for "type": <id>_<type>_prehad.pt   (type in {weight, activation, ...})
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats
import torch
import numpy as np
ALPHA = 0.05
# Padding policy for per-tensor summaries of proportions:
#   - "ignore_pre_padding": use each tensor's own length (default; avoids counting artificial zeros in PRE)
#   - "pad_pre_to_post":    right-pad PRE with zeros to POST length (faithful to transform ambient dim)
#   - "match_min":          compute on the common min length
PAD_MODE_DEFAULT = "ignore_pre_padding"


# --------------------------- I/O + pairing ---------------------------

def load_tensor(filepath: str) -> torch.Tensor:
    """Load a .pt file and return a torch.Tensor (CPU)."""
    obj = torch.load(filepath, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    raise TypeError(f"{filepath} did not contain a torch.Tensor (got {type(obj)}).")


def find_pairs(directory: str) -> List[Tuple[Path, Path]]:
    """Find *_prehad.pt -> *_posthad.pt pairs in the directory (non-recursive)."""
    dir_path = Path(directory)
    pre_files = list(dir_path.glob("*_prehad.pt"))
    pairs: List[Tuple[Path, Path]] = []
    for pre in pre_files:
        post = Path(str(pre).replace("_prehad.pt", "_posthad.pt"))
        if post.exists():
            pairs.append((pre, post))
    return pairs


def _align_for_counts(pre: torch.Tensor, post: torch.Tensor, pad_mode: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Align PRE/POST according to padding policy and return flattened CPU tensors.
    Returns (pre_adj, post_adj, pad_fraction_of_post).
    """
    pre_flat = pre.detach().cpu().reshape(-1)
    post_flat = post.detach().cpu().reshape(-1)
    n_pre = pre_flat.numel()
    n_post = post_flat.numel()

    if pad_mode == "ignore_pre_padding":
        pre_adj = pre_flat
        post_adj = post_flat
        pad_frac = (n_post - n_pre) / float(n_post) if n_post > 0 else float("nan")

    elif pad_mode == "pad_pre_to_post":
        if n_pre < n_post:
            pre_adj = torch.zeros(n_post, dtype=pre_flat.dtype)
            pre_adj[:n_pre] = pre_flat
            pad_frac = (n_post - n_pre) / float(n_post) if n_post > 0 else float("nan")
        elif n_pre > n_post:
            pre_adj = pre_flat[:n_post]
            pad_frac = 0.0
        else:
            pre_adj = pre_flat
            pad_frac = 0.0
        post_adj = post_flat
        return pre_adj, post_adj, float(pad_frac)

    elif pad_mode == "match_min":
        L = min(n_pre, n_post)
        pre_adj = pre_flat[:L]
        post_adj = post_flat[:L]
        pad_frac = (n_post - L) / float(n_post) if n_post > 0 else float("nan")

    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    return pre_adj, post_adj, float(pad_frac)


# --------------------------- Per-tensor summary ---------------------------

def inband_fraction_summary(pre_t: torch.Tensor, post_t: torch.Tensor, eps: float, pad_mode: str) -> Dict[str, float]:
    """
    Compute per-tensor fractions inside [-eps, eps] under the chosen padding policy.
    Returns p_pre, p_post, delta = (p_post - p_pre), plus counts.
    """
    pre_adj, post_adj, pad_frac = _align_for_counts(pre_t, post_t, pad_mode)

    pre_np = pre_adj.numpy().ravel()
    post_np = post_adj.numpy().ravel()

    # Ignore non-finite
    m_pre = np.isfinite(pre_np)
    m_post = np.isfinite(post_np)

    pre_vals = np.abs(pre_np[m_pre])
    post_vals = np.abs(post_np[m_post])

    n_pre = pre_vals.size
    n_post = post_vals.size

    if n_pre == 0 or n_post == 0:
        return {
            "n_pre": int(n_pre), "n_post": int(n_post),
            "pre_in": 0, "post_in": 0, "p_pre": float("nan"), "p_post": float("nan"),
            "delta": float("nan"), "pad_fraction_of_post": pad_frac,
        }

    pre_in = int(np.sum(pre_vals <= eps))
    post_in = int(np.sum(post_vals <= eps))
    p_pre = pre_in / float(n_pre)
    p_post = post_in / float(n_post)

    return {
        "n_pre": int(n_pre), "n_post": int(n_post),
        "pre_in": pre_in, "post_in": post_in,
        "p_pre": float(p_pre), "p_post": float(p_post),
        "delta": float(p_post - p_pre),  # H1: delta > 0
        "pad_fraction_of_post": pad_frac,
    }


def collect_per_tensor_deltas(directory: str, eps: float, pad_mode: str):
    """
    For each matched pair, compute per-tensor in-band fractions and delta.
    Returns:
      - rows: list of dicts with id/type/p_pre/p_post/delta/counts
      - by_type: dict[type] -> 1D np.array of deltas
    """
    rows = []
    by_type: Dict[str, List[float]] = {}

    for pre_path, post_path in find_pairs(directory):
        try:
            pre_t = load_tensor(str(pre_path))
            post_t = load_tensor(str(post_path))
        except Exception as e:
            print(f"[WARN] Skipping pair due to load error: {pre_path} / {post_path} :: {e}")
            continue

        # Infer id/type from filename convention "<id>_<type>_prehad.pt"
        stem = pre_path.stem
        parts = stem.split("_")
        file_id = parts[0]
        file_type = parts[1] if len(parts) > 1 else "unknown"

        s = inband_fraction_summary(pre_t, post_t, eps, pad_mode)
        rows.append({
            "id": file_id, "type": file_type,
            **s
        })
        if not math.isnan(s["delta"]):
            by_type.setdefault(file_type, []).append(s["delta"])

    for t in list(by_type.keys()):
        by_type[t] = np.asarray(by_type[t], dtype=float)

    return rows, by_type


# --------------------------- Tests across tensors ---------------------------

def _normality_pvalues(x: np.ndarray) -> Dict[str, Optional[float]]:
    """Return available normality p-values for a 1D array."""
    shapiro_p: Optional[float] = None
    dago_p: Optional[float] = None
    try:
        if x.size >= 3:
            shapiro_p = float(stats.shapiro(x).pvalue)
    except Exception:
        pass
    try:
        if x.size >= 8:
            dago_p = float(stats.normaltest(x).pvalue)
    except Exception:
        pass
    return {"shapiro_p": shapiro_p, "dagostino_p": dago_p}


def _cohen_dz(d: np.ndarray) -> float:
    """Paired effect size on deltas: mean(delta) / sd(delta)."""
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd) if (np.isfinite(sd) and sd > 0) else float("nan")


def paired_increase_test(deltas: np.ndarray, alpha: float = ALPHA) -> Dict[str, object]:
    """
    Primary: Wilcoxon signed-rank on deltas vs 0, one-sided greater (ties dropped).
    Robustness: exact sign test on wins (delta>0).
    Effect size: Cohen's d_z on deltas.
    """
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return {"error": "no valid deltas"}

    # Wilcoxon (paired), one-sided (greater), ties dropped
    try:
        w_stat, w_p = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
        w_stat = float(w_stat); w_p = float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Exact sign test
    n_pos = int(np.sum(x > 0))
    n_neg = int(np.sum(x < 0))
    n_eff = n_pos + n_neg
    binom_p = float(stats.binomtest(n_pos, n_eff, p=0.5, alternative="greater").pvalue) if n_eff > 0 else float("nan")

    # Effect size
    dz = _cohen_dz(x)

    # Normality info (diagnostic only)
    norm = _normality_pvalues(x)

    return {
        "n": int(n),
        "mean_delta": float(np.mean(x)),
        "median_delta": float(np.median(x)),
        "cohen_dz": dz,
        "wilcoxon": {"stat": w_stat, "p_one_sided": w_p, "significant": (w_p < alpha) if np.isfinite(w_p) else False},
        "sign_test": {"wins": n_pos, "losses": n_neg, "p_one_sided": binom_p, "significant": (binom_p < alpha) if np.isfinite(binom_p) else False},
        "diagnostics": {"shapiro_p": norm["shapiro_p"], "dagostino_p": norm["dagostino_p"]},
    }


def _fmt_block(name: str, res: Dict[str, object]) -> str:
    if "error" in res:
        return f"{name}: ERROR: {res['error']}"
    w = res["wilcoxon"]; s = res["sign_test"]; d = res["diagnostics"]
    return (
        f"{name} (n={res['n']}):\n"
        f"  mean Δ= {res['mean_delta']:.4f}   median Δ= {res['median_delta']:.4f}   Cohen's d_z= {res['cohen_dz']:.3f}\n"
        f"  Wilcoxon (one-sided, greater): stat={w['stat']:.4f}, p={w['p_one_sided']:.6g}, significant={w['significant']}\n"
        f"  Sign test: wins={s['wins']}, losses={s['losses']}, p={s['p_one_sided']:.6g}, significant={s['significant']}\n"
        f"  Normality (diagnostic): Shapiro p={d['shapiro_p']}, D'Agostino p={d['dagostino_p']}"
    )


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Paired test on in-band fractions (|x| <= eps) pre vs post.")
    ap.add_argument("directory", type=str, help="Directory with *_prehad.pt and *_posthad.pt files.")
    ap.add_argument("--eps", type=float, required=True, help="Epsilon defining the in-band region [-eps, eps].")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="Significance level (default 0.05).")
    ap.add_argument("--by-type", action="store_true", help="Also report per-type results (type parsed from filename).")
    ap.add_argument("--pad-mode", type=str, choices=["ignore_pre_padding", "pad_pre_to_post", "match_min"],
                    default=PAD_MODE_DEFAULT, help="Padding policy for per-tensor summaries (default: ignore_pre_padding).")
    args = ap.parse_args()

    rows, by_type = collect_per_tensor_deltas(args.directory, args.eps, args.pad_mode)

    if len(rows) == 0:
        print("No matched *_prehad.pt → *_posthad.pt pairs found.")
        return

    # Global across all tensors
    deltas_all = np.asarray([r["delta"] for r in rows if not math.isnan(r["delta"])], dtype=float)
    res_all = paired_increase_test(deltas_all, alpha=args.alpha)
    print(_fmt_block("\nGLOBAL: in-band fraction increase (post - pre)", res_all))

    # Per-type
    if args.by_type:
        for t, d in by_type.items():
            res_t = paired_increase_test(d, alpha=args.alpha)
            print(_fmt_block(f"\nTYPE = {t}: in-band fraction increase (post - pre)", res_t))
   
    # Optional quick summary table
    n_pairs = len(deltas_all)
    n_inc = int(np.sum(deltas_all > 0))
    n_dec = int(np.sum(deltas_all < 0))
    print(f"\nSummary: total pairs={n_pairs} | increased={n_inc} | decreased={n_dec} | unchanged={n_pairs - n_inc - n_dec}")

if __name__ == "__main__":
    main()
