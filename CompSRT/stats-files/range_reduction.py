import os
import torch
import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Parameters
ALPHA = 0.05                  # Significance level
ALT = 'greater'               # Test direction: pre > post (i.e., reduction after Hadamard)
PAD_MODE = "pad_pre_to_post"  # "pad_pre_to_post" (default) | "ignore_pre_padding" | "match_min"

# ---------------- I/O ----------------

def load_tensor(filepath: str) -> torch.Tensor:
    """Load a .pt file and return the tensor (CPU)."""
    return torch.load(filepath, map_location='cpu')

# ------------- Core range stats (on 1D numpy arrays) -------------

def _range_stats_from_1d(data_1d: np.ndarray) -> Dict[str, float]:
    """Compute range + summary stats on a 1D numpy array, ignoring non-finite values."""
    data = np.asarray(data_1d).ravel()
    finite = np.isfinite(data)
    finite_data = data[finite]
    if finite_data.size == 0:
        return dict(
            range=np.nan, min=np.nan, max=np.nan, mean=np.nan, std=np.nan,
            q1=np.nan, q3=np.nan, iqr=np.nan, total_count=data.size, finite_count=0
        )
    dmin = np.min(finite_data)
    dmax = np.max(finite_data)
    drng = dmax - dmin
    dmean = np.mean(finite_data)
    dstd = np.std(finite_data, ddof=1) if finite_data.size > 1 else 0.0
    q1 = np.percentile(finite_data, 25)
    q3 = np.percentile(finite_data, 75)
    iqr = q3 - q1
    return dict(
        range=drng, min=dmin, max=dmax, mean=dmean, std=dstd,
        q1=q1, q3=q3, iqr=iqr, total_count=data.size, finite_count=finite_data.size
    )

def calculate_range_statistics(tensor: torch.Tensor) -> Dict[str, float]:
    """Compatibility helper; calls the numpy version."""
    return _range_stats_from_1d(tensor.detach().cpu().numpy().ravel())

# --------- Experimental unit = ENTIRE TENSOR (matrix/filter) ----------

def analyze_range_pair(pre_tensor: torch.Tensor, post_tensor: torch.Tensor) -> Dict:
    """
    Experimental unit = the entire tensor (matrix/filter). We summarize PRE and POST
    on the SAME tensor (no channel splitting). Because Hadamard implementations often
    zero-pad inputs to the next power-of-two, we align the ambient dimension via PAD_MODE:

      - "pad_pre_to_post" (default): flatten both; zero-pad PRE (right-pad) to len(POST).
        This makes PRE and POST summaries comparable in the same space the transform used.
      - "ignore_pre_padding": summarize PRE on its original length, POST on its own length.
        (Not recommended—compares different ambient sizes.)
      - "match_min": summarize both on the minimum length (truncate PRE if longer, and
        truncate POST to min length). This discards info from POST’s padded tail.

    We then compute the RANGE = max - min and related quantities on the flattened arrays.
    """
    pre_flat = pre_tensor.detach().cpu().numpy().ravel()
    post_flat = post_tensor.detach().cpu().numpy().ravel()

    n_pre = pre_flat.size
    n_post = post_flat.size

    if PAD_MODE == "pad_pre_to_post":
        if n_pre < n_post:
            # Right-pad PRE with zeros to match POST length
            pre_adj = np.zeros(n_post, dtype=pre_flat.dtype)
            pre_adj[:n_pre] = pre_flat
            pad_fraction = (n_post - n_pre) / float(n_post)
        elif n_pre > n_post:
            # Defensive: clip PRE down to POST length (rare)
            pre_adj = pre_flat[:n_post]
            pad_fraction = 0.0
        else:
            pre_adj = pre_flat
            pad_fraction = 0.0
        post_adj = post_flat

    elif PAD_MODE == "match_min":
        L = min(n_pre, n_post)
        pre_adj = pre_flat[:L]
        post_adj = post_flat[:L]
        pad_fraction = max(n_post - L, 0) / float(n_post) if n_post > 0 else np.nan

    elif PAD_MODE == "ignore_pre_padding":
        pre_adj = pre_flat
        post_adj = post_flat
        pad_fraction = max(n_post - n_pre, 0) / float(n_post) if n_post > 0 else np.nan

    else:
        raise ValueError(f"Unknown PAD_MODE: {PAD_MODE}")

    pre_stats = _range_stats_from_1d(pre_adj)
    post_stats = _range_stats_from_1d(post_adj)

    # Delta definitions (positive means reduction)
    range_delta = pre_stats['range'] - post_stats['range']
    if np.isfinite(pre_stats['range']) and pre_stats['range'] > 0:
        rel_reduction = (range_delta / pre_stats['range']) * 100.0
    else:
        rel_reduction = 0.0 if post_stats['range'] == 0 else np.nan

    # Normalized ranges (range / std), defensive
    pre_norm = pre_stats['range'] / pre_stats['std'] if (pre_stats['std'] and np.isfinite(pre_stats['std']) and pre_stats['std'] > 0) else np.nan
    post_norm = post_stats['range'] / post_stats['std'] if (post_stats['std'] and np.isfinite(post_stats['std']) and post_stats['std'] > 0) else np.nan

    return {
        'pre_stats': pre_stats,
        'post_stats': post_stats,
        'range_reduction': range_delta,
        'relative_reduction': rel_reduction,
        'range_decreased': (post_stats['range'] < pre_stats['range']) if (np.isfinite(post_stats['range']) and np.isfinite(pre_stats['range'])) else False,
        'pre_normalized_range': pre_norm,
        'post_normalized_range': post_norm,
        'pad_fraction_of_post': pad_fraction,
        'n_pre': int(n_pre),
        'n_post': int(n_post),
        'pad_mode': PAD_MODE,
    }

# ------------- Directory scan (one row per TENSOR file) -------------

def analyze_directory(directory: str) -> pd.DataFrame:
    """
    Find *_prehad.pt → *_posthad.pt pairs (non-recursive).
    ONE ROW PER TENSOR (entire file tensor = experimental unit).
    """
    dir_path = Path(directory)
    prehad_files = list(dir_path.glob("*_prehad.pt"))

    results = []
    for prehad_file in prehad_files:
        posthad_file = prehad_file.with_name(prehad_file.name.replace('_prehad.pt', '_posthad.pt'))
        if not posthad_file.exists():
            print(f"Warning: No matching post-HAD file for {prehad_file}")
            continue

        # Infer type/id from filename (kept as-is; your naming may differ)
        parts = prehad_file.stem.split('_')
        file_id = parts[0]
        file_type = parts[1] if len(parts) > 1 else 'unknown'

        try:
            pre_tensor = load_tensor(str(prehad_file))
            post_tensor = load_tensor(str(posthad_file))

            print(f"Processing {file_id}_{file_type}: pre shape={tuple(pre_tensor.shape)}, post shape={tuple(post_tensor.shape)}")

            analysis = analyze_range_pair(pre_tensor, post_tensor)

            results.append({
                'id': file_id,
                'type': file_type,
                'pre_range': analysis['pre_stats']['range'],
                'post_range': analysis['post_stats']['range'],
                'range_reduction': analysis['range_reduction'],
                'relative_reduction': analysis['relative_reduction'],
                'range_decreased': analysis['range_decreased'],
                'pre_normalized_range': analysis['pre_normalized_range'],
                'post_normalized_range': analysis['post_normalized_range'],
                'pad_fraction_of_post': analysis['pad_fraction_of_post'],
                'n_pre': analysis['n_pre'],
                'n_post': analysis['n_post'],
                'pad_mode': analysis['pad_mode'],
            })

        except Exception as e:
            print(f"Error processing {prehad_file}: {e}")

    return pd.DataFrame(results)

# ------------- Paired tests across tensors (per type) -------------

def _normality_pvalues(x: np.ndarray) -> Tuple[float, float]:
    """Return (shapiro_p, dagostino_p) if applicable, else (None, None)."""
    shapiro_p = None
    dagostino_p = None
    try:
        if x.size >= 3:
            shapiro_p = float(stats.shapiro(x).pvalue)
    except Exception:
        pass
    try:
        if x.size >= 8:
            dagostino_p = float(stats.normaltest(x).pvalue)
    except Exception:
        pass
    return shapiro_p, dagostino_p

def _decide_normal(shapiro_p, dagostino_p, alpha=ALPHA) -> bool:
    decisions = []
    if shapiro_p is not None: decisions.append(shapiro_p >= alpha)
    if dagostino_p is not None: decisions.append(dagostino_p >= alpha)
    return (len(decisions) > 0) and all(decisions)

def _cohen_dz(d: np.ndarray) -> float:
    """Matched-pairs Cohen's d_z on paired differences."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float('nan')
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if (np.isfinite(sd) and sd > 0) else float('nan')

def paired_tensor_summary_tests(df: pd.DataFrame, tensor_type: str) -> Dict:
    """
    Build paired arrays (pre_range, post_range) across tensors of the same type,
    then test whether pre > post (ALT='greater').

    Primary: Wilcoxon signed-rank (ties dropped, one-sided).
    Robustness: exact sign test on signs of (pre - post).
    Optional: paired t-test when deltas look normal.
    """
    type_df = df[df['type'] == tensor_type].copy()
    if len(type_df) < 2:
        return {'error': f'Insufficient {tensor_type} samples for testing (n={len(type_df)})'}

    pre = type_df['pre_range'].to_numpy(dtype=float)
    post = type_df['post_range'].to_numpy(dtype=float)

    valid = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[valid], post[valid]
    if pre.size < 2:
        return {'error': f'Insufficient valid {tensor_type} pairs for testing'}

    deltas = pre - post

    # Normality on paired differences
    shapiro_p, dago_p = _normality_pvalues(deltas)
    is_normal = _decide_normal(shapiro_p, dago_p, alpha=ALPHA)

    # Primary: Wilcoxon signed-rank (paired), one-sided, ties dropped
    try:
        w_stat, w_p = stats.wilcoxon(pre, post, alternative=ALT, zero_method='wilcox')
        w_stat = float(w_stat); w_p = float(w_p)
    except Exception:
        w_stat, w_p = float('nan'), float('nan')

    # Optional: paired t-test one-sided (if normal)
    try:
        t_res = stats.ttest_rel(pre, post, alternative=ALT)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    except TypeError:
        t_stat2, p_two = stats.ttest_rel(pre, post)
        if np.isnan(t_stat2) or np.isnan(p_two):
            t_stat, t_p = float('nan'), float('nan')
        else:
            t_stat = float(t_stat2)
            t_p = (p_two / 2.0) if t_stat > 0 else (1.0 - p_two / 2.0)

    # Exact sign test (binomial on direction of deltas)
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_tie = int(np.sum(deltas == 0))
    n_eff = n_pos + n_neg  # ties excluded
    binom_p = float(stats.binomtest(n_pos, n_eff, p=0.5, alternative='greater').pvalue) if n_eff > 0 else float('nan')

    # Effect size on paired deltas
    dz = _cohen_dz(deltas)

    return {
        'n_pairs': int(pre.size),
        'summary': {
            'mean_pre_range': float(np.nanmean(pre)),
            'mean_post_range': float(np.nanmean(post)),
            'median_pre_range': float(np.nanmedian(pre)),
            'median_post_range': float(np.nanmedian(post)),
            'mean_delta': float(np.mean(deltas)),
            'median_delta': float(np.median(deltas)),
            'cohen_dz': dz
        },
        'wilcoxon': {'stat': w_stat, 'p_one_sided': w_p, 'significant': (w_p < ALPHA) if np.isfinite(w_p) else False},
        'paired_t': {'stat': t_stat, 'p_one_sided': t_p, 'significant': (t_p < ALPHA) if np.isfinite(t_p) else False,
                     'normal_deltas': is_normal, 'shapiro_p': shapiro_p, 'dagostino_p': dago_p},
        'sign_test': {'n_pos': n_pos, 'n_neg': n_neg, 'n_tie': n_tie, 'p_one_sided': binom_p, 'significant': (binom_p < ALPHA) if np.isfinite(binom_p) else False}
    }

# ------------- Main -------------

def main(directory: str):
    """
    Experimental unit = entire tensor (matrix/filter).
    We summarize each tensor pre/post (with PAD_MODE policy) and then run paired tests
    across tensors of the same type.
    """
    print(f"Analyzing range reduction in directory: {directory}")
    print("Elements are not individually matchable across Hadamard; post entries are linear combinations of many pre entries.")
    print("Experimental unit = entire tensor (matrix/filter); no per-channel splitting.")
    print(f"Padding policy: PAD_MODE='{PAD_MODE}' (default pads PRE to POST length).")
    print(f"Summary metric: RANGE = max - min")
    print(f"Test direction: {ALT} (pre > post)")
    print(f"Significance level: α = {ALPHA}")
    print("=" * 72)

    df = analyze_directory(directory)
    if df.empty:
        print("No valid pre/post HAD pairs found!")
        return

    print(f"\nFound {len(df)} paired tensors")
    print(f"Types: {df['type'].value_counts().to_dict()}")

    for tensor_type in ['weight', 'activation']:
        if tensor_type in df['type'].values:
            print("\n" + "─" * 72)
            print(f"{tensor_type.upper()}S: paired tests across tensors")
            print("─" * 72)
            res = paired_tensor_summary_tests(df, tensor_type)
            if 'error' in res:
                print(res['error']); continue

            s = res['summary']
            print(f"n = {res['n_pairs']} tensors")
            print(f"Pre vs Post RANGE (mean / median): {s['mean_pre_range']:.6f} / {s['median_pre_range']:.6f}  →  {s['mean_post_range']:.6f} / {s['median_post_range']:.6f}")
            print(f"Delta (pre - post): mean={s['mean_delta']:.6f}, median={s['median_delta']:.6f}")
            print(f"Effect size (Cohen's d_z on paired deltas): {s['cohen_dz']:.3f}")  # <-- printed effect size

            w = res['wilcoxon']
            print(f"\nWilcoxon signed-rank (one-sided, ties dropped): stat={w['stat']:.4f}, p={w['p_one_sided']:.6g}, significant={w['significant']}")

            t = res['paired_t']
            print(f"Paired t-test (one-sided): stat={t['stat']:.4f}, p={t['p_one_sided']:.6g}, significant={t['significant']}")
            print(f"Normality of deltas — Shapiro p={t['shapiro_p']}, D’Agostino p={t['dagostino_p']}, approx normal? {t['normal_deltas']}")

            sg = res['sign_test']
            print(f"Exact sign test (binomial, ties excluded): wins={sg['n_pos']}, losses={sg['n_neg']}, ties={sg['n_tie']}, p={sg['p_one_sided']:.6g}, significant={sg['significant']}")

    # Optional overview
    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)
    all_decreased = int(df['range_decreased'].sum())
    all_total = int(len(df))
    print(f"\nAcross all {all_total} tensors:")
    print(f"  Tensors with reduced range: {all_decreased} ({all_decreased/all_total*100:.1f}%)")

if __name__ == "__main__":
    # Set your directory path here
    directory_path = "path/to/weights_and_activs"  # <-- change this
    main(directory_path)
