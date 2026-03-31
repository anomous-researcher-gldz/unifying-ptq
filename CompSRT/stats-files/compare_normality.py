#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_normality_rrstyle.py

Mirrors the I/O/structure style of `range_reduction_final.py`:
- Parameters defined at the top (ALPHA, PAD_MODE)
- A `main(directory_path)` entry point
- Directory path set at bottom of file (edit to your own path)
- Scans directory for pre/post tensor pairs and aggregates results into a DataFrame

Task:
  Compare how normally distributed tensors are "pre" vs "post" Hadamard.
  Compute multiple normality metrics per tensor, then test whether POST is
  closer to normal than PRE using paired one-sided Wilcoxon on metric deltas.

Filename conventions supported (both directions):
  *.pre.pt <-> *.post.pt
  *_pre.pt <-> *_post.pt

Padding/shape handling (PAD_MODE):
  - "pad_pre_to_post": if sizes differ, pad PRE with zeros to POST size (use with caution)
  - "ignore_pre_padding": if a known padding value (0.0) exists at the tail of PRE,
    drop tail zeros from PRE before comparison (no growth beyond POST size).
  - "match_min": use the common length = min(len(pre), len(post)) by truncation (recommended)

Outputs:
  - CSV: normality_pairs_metrics.csv (in the working directory)
  - Printed summaries (overall and by inferred type: weights/activations)

Edit the `directory_path` at the bottom and run:
  python compare_normality_rrstyle.py
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings('ignore')

# ---------------- Parameters ----------------
ALPHA = 0.05
PAD_MODE = "match_min"   # "pad_pre_to_post" | "ignore_pre_padding" | "match_min"
SAVE_CSV = "normality_pairs_metrics.csv"

# ---------------- Utilities ----------------

SUFFIXES = [
    (r'([_\-.])prehad\.pt$',  r'\1posthad.pt'),
    (r'([_\-.])posthad\.pt$', r'\1prehad.pt'),
    (r'([_\-.])pre\.pt$',     r'\1post.pt'),
    (r'([_\-.])post\.pt$',    r'\1pre.pt'),
]
def is_pre_file(p: Path) -> bool:
    s = str(p)
    # Accept *_pre.pt, *_prehad.pt, and dash/dot variants like -pre.pt or .prehad.pt
    return bool(re.search(r'(?:^|[_\-.])pre(?:had)?\.pt$', s))

def find_post_match(pre_path: Path) -> Optional[Path]:
    s = str(pre_path)
    for pat, repl in SUFFIXES:
        if re.search(pat, s):
            cand = Path(re.sub(pat, repl, s))
            if cand.exists():
                return cand
    return None

def infer_type_from_name(name: str) -> str:
    lname = name.lower()
    if 'act' in lname or 'activation' in lname:
        return 'activations'
    if 'weight' in lname or 'weights' in lname or 'wgt' in lname:
        return 'weights'
    return 'unknown'

def to_1d_float(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()

def align_arrays(pre: np.ndarray, post: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    n_pre, n_post = pre.size, post.size
    if n_pre == n_post:
        return pre, post

    if mode == "match_min":
        n = min(n_pre, n_post)
        return pre[:n], post[:n]

    if mode == "ignore_pre_padding":
        # Drop trailing zeros from PRE if it is longer; otherwise fall back to match_min.
        if n_pre > n_post:
            # remove trailing zeros only
            nz = np.nonzero(pre != 0.0)[0]
            last_nz = nz[-1] + 1 if nz.size else 0
            trimmed = pre[:max(last_nz, n_post)]
            n = min(trimmed.size, n_post)
            return trimmed[:n], post[:n]
        else:
            n = min(n_pre, n_post)
            return pre[:n], post[:n]

    if mode == "pad_pre_to_post":
        if n_pre < n_post:
            pad = np.zeros(n_post - n_pre, dtype=pre.dtype)
            pre2 = np.concatenate([pre, pad], axis=0)
            return pre2, post
        else:
            # if pre is longer, fallback to match_min (avoid padding post with zeros which changes dist)
            n = min(n_pre, n_post)
            return pre[:n], post[:n]

    # default safety
    n = min(n_pre, n_post)
    return pre[:n], post[:n]

# ---------------- Normality metrics ----------------

def safe_sample(x: np.ndarray, max_n: int) -> np.ndarray:
    if x.size <= max_n:
        return x
    rng = np.random.default_rng(123)
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]

def compute_metrics(x: np.ndarray, shapiro_n: int = 5000) -> Dict[str, float]:
    """
    Compute multiple normality-related metrics on 1D array x.
    Returns:
      - w_shapiro, p_shapiro
      - k2, p_k2
      - ad  (Anderson-Darling statistic for normal)
      - jb, p_jb
      - skew, kurt_excess
    """
    out: Dict[str, float] = {}

    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        keys = ['w_shapiro','p_shapiro','k2','p_k2','ad','jb','p_jb','skew','kurt_excess']
        for k in keys:
            out[k] = np.nan
        out['n'] = n
        return out

    # Shapiro on up to 5000 samples
    x_sh = safe_sample(x, 1000000)
    try:
        w, p = stats.shapiro(x_sh)
    except Exception:
        w, p = np.nan, np.nan
    out['w_shapiro'] = float(w)
    out['p_shapiro'] = float(p)

    # D'Agostino K^2
    try:
        k2, p_k2 = stats.normaltest(x_sh)
    except Exception:
        k2, p_k2 = np.nan, np.nan
    out['k2'] = float(k2)
    out['p_k2'] = float(p_k2)

    # Anderson-Darling
    try:
        ad_res = stats.anderson(x_sh, dist='norm')
        ad = float(ad_res.statistic)
    except Exception:
        ad = np.nan
    out['ad'] = ad

    # Jarque-Bera
    try:
        jb, p_jb = stats.jarque_bera(x_sh)
    except Exception:
        jb, p_jb = np.nan, np.nan
    out['jb'] = float(jb)
    out['p_jb'] = float(p_jb)

    # Skew & excess kurtosis
    try:
        skew = stats.skew(x_sh, bias=False)
    except Exception:
        skew = np.nan
    try:
        kurt_excess = stats.kurtosis(x_sh, fisher=True, bias=False)
    except Exception:
        kurt_excess = np.nan
    out['skew'] = float(skew)
    out['kurt_excess'] = float(kurt_excess)
    out['n'] = int(n)
    return out

def wilcoxon_one_sided_greater(deltas: np.ndarray):
    deltas = deltas[np.isfinite(deltas)]
    deltas = deltas[deltas != 0]
    n = deltas.size
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    res = stats.wilcoxon(deltas, alternative='greater', zero_method='wilcox', correction=False, method='approx')
    W = float(res.statistic)
    p = float(res.pvalue)
    try:
        z = stats.norm.isf(p)  # one-sided
        r = float(z / np.sqrt(n))
    except Exception:
        r = np.nan
    return (W, p, r, n)

# ---------------- Core processing ----------------

def main(directory_path: str):
    root = Path(directory_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    # Collect files
    files: List[Path] = list(root.rglob("*.pt"))
    pre_files = [p for p in files if is_pre_file(p)]

    pairs: List[Tuple[Path, Path]] = []
    for pre in pre_files:
        post = find_post_match(pre)
        if post is not None and post.exists():
            pairs.append((pre, post))

    if not pairs:
        print("No pre/post pairs found. Adjust filename conventions if needed.")
        return

    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(123)

    for pre, post in pairs:
        try:
            x_pre = torch.load(pre, map_location="cpu")
            x_post = torch.load(post, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Failed to load pair {pre} / {post}: {e}")
            continue

        x_pre = to_1d_float(x_pre)
        x_post = to_1d_float(x_post)

        # Align lengths according to PAD_MODE
        x_pre, x_post = align_arrays(x_pre, x_post, PAD_MODE)

        # Compute metrics
        m_pre = compute_metrics(x_pre)
        m_post = compute_metrics(x_post)

        # Type inference & row assembly
        inferred_type = infer_type_from_name(pre.name) or infer_type_from_name(post.name)
        row = {
            "pair": str(pre),
            "type": inferred_type,
            "n_pre": m_pre["n"],
            "n_post": m_post["n"],

            "pre_w": m_pre["w_shapiro"], "pre_p_shapiro": m_pre["p_shapiro"],
            "post_w": m_post["w_shapiro"], "post_p_shapiro": m_post["p_shapiro"],

            "pre_k2": m_pre["k2"], "pre_p_k2": m_pre["p_k2"],
            "post_k2": m_post["k2"], "post_p_k2": m_post["p_k2"],

            "pre_ad": m_pre["ad"], "post_ad": m_post["ad"],
            "pre_jb": m_pre["jb"], "pre_p_jb": m_pre["p_jb"],
            "post_jb": m_post["jb"], "post_p_jb": m_post["p_jb"],

            "pre_skew": m_pre["skew"], "post_skew": m_post["skew"],
            "pre_kurt_excess": m_pre["kurt_excess"], "post_kurt_excess": m_post["kurt_excess"],
        }

        # Deltas oriented so that positive = post more normal
        row["delta_W"]  = (m_post["w_shapiro"] - m_pre["w_shapiro"]) if np.isfinite(m_post["w_shapiro"]) and np.isfinite(m_pre["w_shapiro"]) else np.nan
        row["delta_K2"] = (m_pre["k2"] - m_post["k2"])               if np.isfinite(m_post["k2"]) and np.isfinite(m_pre["k2"]) else np.nan
        row["delta_AD"] = (m_pre["ad"] - m_post["ad"])               if np.isfinite(m_post["ad"]) and np.isfinite(m_pre["ad"]) else np.nan
        row["delta_JB"] = (m_pre["jb"] - m_post["jb"])               if np.isfinite(m_post["jb"]) and np.isfinite(m_pre["jb"]) else np.nan

        rows.append(row)

    if not rows:
        print("No valid rows to analyze.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(SAVE_CSV, index=False)
    print(f"Saved per-pair metrics to {SAVE_CSV}  (pairs={len(df)})")

    # ---------------- Summaries (overall & by type) ----------------
    def summarize(block: pd.DataFrame, label: str):
        print("\n" + "=" * 72)
        print(f"SUMMARY: {label} (pairs={len(block)})")
        print("=" * 72)

        def run(metric_name: str, deltas: pd.Series):
            arr = deltas.to_numpy(dtype=float)
            improved = int(np.sum(arr > 0))
            worsened = int(np.sum(arr < 0))
            same = int(np.sum(arr == 0))
            W, p, r, n_used = wilcoxon_one_sided_greater(arr)
            print(f"{metric_name:>10s}: improved={improved}, worsened={worsened}, same={same}, used={n_used}")
            print(f"             Wilcoxon one-sided (Δ>0): W={W:.3f}  p={p:.3e}  effect_size_r={r:.3f}")

        run("Δ_W (↑)",  block["delta_W"].where(np.isfinite(block["delta_W"])).dropna())
        run("Δ_K2 (↑)", block["delta_K2"].where(np.isfinite(block["delta_K2"])).dropna())
        run("Δ_AD (↑)", block["delta_AD"].where(np.isfinite(block["delta_AD"])).dropna())
        run("Δ_JB (↑)", block["delta_JB"].where(np.isfinite(block["delta_JB"])).dropna())

        # Simple binomial sign test (how often post improved)
        for col, label2 in [("delta_W","W higher"), ("delta_K2","K2 lower"), ("delta_AD","AD lower"), ("delta_JB","JB lower")]:
            arr = block[col].to_numpy(dtype=float)
            k = int(np.sum(arr > 0))
            n = int(np.sum(np.isfinite(arr)))
            if n > 0:
                p_sign = stats.binomtest(k, n, 0.5, alternative='greater').pvalue
                print(f"   Sign test {label2:>8s}: k={k}/{n}  p={p_sign:.3e}")

    summarize(df, "ALL TYPES")
    for t in sorted(df["type"].dropna().unique()):
        summarize(df[df["type"] == t], f"type = {t}")

    print("\nNotes:")
    print(" - Positive deltas mean POST is closer to normal than PRE for that metric.")
    print(" - Shapiro W: higher is better; K2/AD/JB: lower is better.")
    print(f" - PAD_MODE = '{PAD_MODE}'. For shape mismatches, this controls alignment.")
    print(" - Effect size r = z/sqrt(n) for Wilcoxon; also prints a sign test as a robustness check.")
    print(" - Edit the 'directory_path' at the bottom of the file to point at your data.\n")

# ---------------- Entry point ----------------

if __name__ == "__main__":
    # Set your directory path here (mirror style of range_reduction_final.py)
    directory_path = "path/to/weights_and_activs"  # <-- change this
    main(directory_path)
