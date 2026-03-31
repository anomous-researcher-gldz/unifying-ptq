#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paired_ds_from_dict_min.py
==========================

Reads the local `scores` dict and, for each (scale, bit, metric in {PSNR, SSIM}),
prints ONLY:
  • Wilcoxon one-sided p-value (A > B by default)
  • Cohen's d_z (paired effect size on deltas)

Run:
    python paired_ds_from_dict_min.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats

def holm_bonferroni(pvals, alpha=0.05):
    """
    Holm step-down adjustment for a list/array of p-values (one-sided or two-sided).
    Returns adjusted p-values in the original order and reject decisions.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty_like(p)
    running_max = 0.0
    for k, idx in enumerate(order, start=1):
        mult = (m - k + 1)
        val = p[idx] * mult
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    # Rejections with Holm’s sequential rule
    reject = np.zeros(m, dtype=bool)
    thresh = alpha / (m - np.arange(m))  # alpha/(m-k+1) in original order
    # Need sequential check in sorted order
    passed = True
    for k, idx in enumerate(order):
        if passed and p[idx] <= alpha / (m - k):
            reject[idx] = True
        else:
            passed = False
            reject[idx] = False
    return adj, reject

def dz_and_ci_onesided(deltas, alpha=0.05, alternative="greater"):
    """
    Paired Cohen's d_z and its 95% one-sided CI.
    d_z = mean(delta)/sd(delta).  n = len(delta).
    For one-sided 95%:
      - 'greater':  [d_z - t_{.95}/sqrt(n),  +inf)
      - 'less':     (-inf,  d_z + t_{.95}/sqrt(n)]
    """
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan, (np.nan, np.nan)
    s = x.std(ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.nan, (np.nan, np.nan)
    dz = x.mean() / s
    tcrit = stats.t.ppf(1 - alpha, df=n - 1)  # one-sided
    half = tcrit / np.sqrt(n)
    if alternative == "greater":
        ci = (dz - half, np.inf)
    elif alternative == "less":
        ci = (-np.inf, dz + half)
    else:
        # two-sided fallback
        t2 = stats.t.ppf(1 - alpha/2, df=n - 1)
        ci = (dz - t2/np.sqrt(n), dz + t2/np.sqrt(n))
    return float(dz), (float(ci[0]), float(ci[1]))

# ------------------------------ CONFIG -------------------------------------

A_NAME = "SRTQuant"   # left element of each tuple in `scores`
B_NAME = "CondiQuant" # right element
ALTERNATIVE = "greater"  # "greater" tests A > B

# ------------------------------ DATA ---------------------------------------

@dataclass
class Scores:
    psnr: float
    ssim: float

# Dict[(scale:str, bit:int)] -> Dict[dataset:str] -> (Scores_A, Scores_B)
scores: Dict[Tuple[str, int], Dict[str, Tuple[Scores, Scores]]] = {
    # =============== x2 ===============
    ("x2", 4): {
        "Set5"    : (Scores(38.13, 0.9610), Scores(38.03, 0.9605)),
        "Set14"   : (Scores(33.81, 0.9203), Scores(33.50, 0.9180)),
        "B100"    : (Scores(32.28, 0.9009), Scores(32.16, 0.8993)),
        "Urban100": (Scores(32.57, 0.9325), Scores(32.03, 0.9282)),
        "Manga109": (Scores(38.98, 0.9778), Scores(38.57, 0.9769)),
    },
    ("x2", 3): {
        "Set5"    : (Scores(38.11, 0.9609), Scores(37.77, 0.9594)),
        "Set14"   : (Scores(33.82, 0.9202), Scores(33.21, 0.9151)),
        "B100"    : (Scores(32.27, 0.9008), Scores(31.94, 0.8966)),
        "Urban100": (Scores(32.53, 0.9321), Scores(31.18, 0.9197)),
        "Manga109": (Scores(38.90, 0.9775), Scores(38.01, 0.9755)),
    },
    ("x2", 2): {
        "Set5"    : (Scores(38.03, 0.9605), Scores(37.15, 0.9567)),
        "Set14"   : (Scores(33.70, 0.9194), Scores(32.74, 0.9103)),
        "B100"    : (Scores(32.19, 0.9294), Scores(31.55, 0.8912)),
        "Urban100": (Scores(32.22, 0.9294), Scores(29.96, 0.9047)),
        "Manga109": (Scores(38.69, 0.9770), Scores(36.63, 0.9713)),
    },

    # =============== x3 ===============
    ("x3", 4): {
        "Set5"    : (Scores(34.56, 0.9284), Scores(34.32, 0.9260)),
        "Set14"   : (Scores(30.49, 0.8454), Scores(30.29, 0.8417)),
        "B100"    : (Scores(29.17, 0.8075), Scores(29.05, 0.8039)),
        "Urban100": (Scores(28.50, 0.8598), Scores(28.05, 0.8506)),
        "Manga109": (Scores(33.83, 0.9467), Scores(33.23, 0.9431)),
    },
    ("x3", 3): {
        "Set5"    : (Scores(34.54, 0.9281), Scores(33.92, 0.9224)),
        "Set14"   : (Scores(30.48, 0.8451), Scores(30.02, 0.8367)),
        "B100"    : (Scores(29.16, 0.8070), Scores(28.84, 0.7986)),
        "Urban100": (Scores(28.47, 0.8589), Scores(27.37, 0.8356)),
        "Manga109": (Scores(33.79, 0.9465), Scores(32.48, 0.9367)),
    },
    ("x3", 2): {
        "Set5"    : (Scores(34.17, 0.9248), Scores(33.00, 0.9130)),
        "Set14"   : (Scores(30.21, 0.8401), Scores(29.44, 0.8253)),
        "B100"    : (Scores(28.97, 0.8017), Scores(28.45, 0.7882)),
        "Urban100": (Scores(27.86, 0.8456), Scores(26.36, 0.8080)),
        "Manga109": (Scores(33.11, 0.9414), Scores(30.88, 0.9203)),
    },

    # =============== x4 ===============
    ("x4", 4): {
        "Set5"    : (Scores(32.41, 0.8969), Scores(32.09, 0.8923)),
        "Set14"   : (Scores(28.74, 0.7849), Scores(28.50, 0.7792)),
        "B100"    : (Scores(27.68, 0.7399), Scores(27.52, 0.7345)),
        "Urban100": (Scores(26.39, 0.7953), Scores(25.97, 0.7831)),
        "Manga109": (Scores(30.81, 0.9131), Scores(30.16, 0.9054)),
    },
    ("x4", 3): {
        "Set5"    : (Scores(32.31, 0.8956), Scores(31.62, 0.8855)),
        "Set14"   : (Scores(28.69, 0.7839), Scores(28.20, 0.7715)),
        "B100"    : (Scores(27.64, 0.7387), Scores(27.31, 0.7269)),
        "Urban100": (Scores(26.27, 0.7918), Scores(25.39, 0.7624)),
        "Manga109": (Scores(30.60, 0.9108), Scores(29.29, 0.8915)),
    },
    ("x4", 2): {
        "Set5"    : (Scores(31.44, 0.8820), Scores(30.64, 0.8671)),
        "Set14"   : (Scores(28.15, 0.7696), Scores(27.59, 0.7567)),
        "B100"    : (Scores(27.28, 0.7253), Scores(26.93, 0.7136)),
        "Urban100": (Scores(25.38, 0.7585), Scores(24.54, 0.7282)),
        "Manga109": (Scores(29.20, 0.8881), Scores(27.67, 0.8613)),
    },
}

# --------------------------- Helpers & Core --------------------------------

def _build_dataframe(scores: Dict[Tuple[str,int], Dict[str, Tuple[Scores, Scores]]]) -> pd.DataFrame:
    rows = []
    for (scale, bit), dsmap in scores.items():
        sc = str(scale); bt = str(bit)
        for ds, (a, b) in dsmap.items():
            rows.append({
                "dataset": ds, "scale": sc, "bit": bt,
                "a_psnr": float(a.psnr), "b_psnr": float(b.psnr),
                "a_ssim": float(a.ssim), "b_ssim": float(b.ssim),
            })
    return pd.DataFrame(rows)

def _cohen_dz(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan")
    sd = np.std(d, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.mean(d) / sd)

def _summarize_min(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, bt), g in df.groupby(["scale","bit"]):
        for metric in ["psnr","ssim"]:
            a = g[f"a_{metric}"].to_numpy(dtype=float)
            b = g[f"b_{metric}"].to_numpy(dtype=float)
            d = a - b
            # Wilcoxon (one-sided, ties dropped)
            try:
                wres = stats.wilcoxon(d, zero_method="wilcox", alternative=ALTERNATIVE, method="auto")
                pval = float(wres.pvalue)
            except Exception:
                pval = float("nan")
            dz = _cohen_dz(d)
            rows.append({
                "scale": sc,
                "bit": bt,
                "alternative": "greater",
                "deltas": d,  
                "n": len(d), 
                "metric": metric.upper(),
                "n_datasets": int(d.size),
                "wilcoxon_p_one_sided": pval,
                "cohen_dz": dz,
            })
    return pd.DataFrame(rows).sort_values(["scale","bit","metric"])

def _print_min(tbl: pd.DataFrame):
    if tbl.empty:
        print("No rows to summarize. Check the `scores` dict.")
        return
    shown = tbl.copy()
    shown["wilcoxon_p_one_sided"] = shown["wilcoxon_p_one_sided"].map(
        lambda x: f"{x:.3g}" if pd.notna(x) else ""
    )
    shown["cohen_dz"] = shown["cohen_dz"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else ""
    )
    cols = ["scale","bit","metric","n_datasets","wilcoxon_p_one_sided","cohen_dz"]
    print(f"=== {A_NAME} vs {B_NAME} | alternative: {ALTERNATIVE} ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(shown[cols].to_string(index=False))

def run():
    df = _build_dataframe(scores)
    # print(df)
    tbl = _summarize_min(df)
    # assert len(df) == 18, "Holm correction should include all 18 tests."
    # tbl["p_holm"], tbl["reject_holm_0.05"] = holm_bonferroni(tbl["wilcoxon_p_one_sided"].values, alpha=0.05)

    # # (B) d_z and 95% one-sided CI per test (direction kept via 'alternative')
    # dz_list, lo_list, hi_list = [], [], []
    # for alt, deltas in zip(tbl["alternative"].values, tbl["deltas"].values):
    #     dz, (lo, hi) = dz_and_ci_onesided(deltas, alpha=0.05, alternative=alt)
    #     dz_list.append(dz); lo_list.append(lo); hi_list.append(hi)

    # tbl["dz"] = dz_list
    # tbl["dz_CI_low"]  = lo_list    # one-sided bound (other side is +/- inf)
    # tbl["dz_CI_high"] = hi_list

    # Optional: pretty print or save
    # cols = [ "alternative", "n", "wilcoxon_p_one_sided", "p_holm", "reject_holm_0.05", "dz", "dz_CI_low", "dz_CI_high"]
    # print(tbl[cols].to_string(index=False))
    _print_min(tbl)

if __name__ == "__main__":
    run()

