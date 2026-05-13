"""Synthetic dense-vs-sparse outlier study (extends §4.3).

Tests the claim: DBAF's fold/unfold is information-preserving when outliers
are sparse (rare tail values), and lossy when outliers are dense (heavy-tailed
body). The MSE may decrease in both regimes, but the *information content*
preserved differs.

We construct two families of distributions with matched 3σ-fraction but
different *concentration*:
  - sparse: Gaussian core (~99%) + handful of outliers far from zero
  - dense:  Heavy-tailed (Student-t with low df) — outliers blend into body

For each, run DBAF fold + INT4 RTN quant + unfold. Measure:
  - MSE of reconstruction (per-tensor)
  - Bit-distinct count (how many unique INT4 codes are used)
  - Reconstruction MSE on the *body* (|x| <= T) and on the *tail* (|x| > T)
    separately. The information-preserving claim: sparse keeps body intact;
    dense distorts the body to preserve range.

Output: per-distribution metrics + a plot.
"""
from __future__ import annotations
import sys, json, pathlib, math
import numpy as np
import torch
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from ahcptq.quantization.fake_quant import fold_outliers, unfold_outliers
from flatquant.baselines.rtn import _quantize_tensor_uniform


def synth_sparse(n: int, k: int, outlier_mag: float, rng) -> torch.Tensor:
    """Standard normal body + k outliers placed at ±outlier_mag."""
    x = rng.normal(0, 1, n)
    idx = rng.choice(n, size=k, replace=False)
    x[idx] = (2 * rng.integers(0, 2, k) - 1) * outlier_mag
    return torch.tensor(x, dtype=torch.float32)


def synth_dense(n: int, df: float, rng) -> torch.Tensor:
    """Student-t with df degrees of freedom — small df = heavy tails."""
    x = rng.standard_t(df, size=n)
    # Normalize so std ~ 1 for fair comparison.
    x = x / (x.std() + 1e-8)
    return torch.tensor(x, dtype=torch.float32)


def evaluate(x: torch.Tensor, bits: int = 4, alpha: float = 0.95) -> dict:
    """Apply DBAF fold + per-channel RTN + unfold; report metrics.

    Treats x as a single row, then uses 1D RTN with symmetric per-tensor scale.
    """
    x = x.unsqueeze(0)  # shape [1, n]
    qmax = 2 ** (bits - 1) - 1
    # No-DBAF baseline: just RTN.
    w_rtn = _quantize_tensor_uniform(x, bits, per_channel=True)
    # DBAF: fold first.
    T = float(3.0 * x.std().clamp_min(1e-8))
    x_fold, tag = fold_outliers(x, T, alpha)
    w_q = _quantize_tensor_uniform(x_fold, bits, per_channel=True)
    x_dbaf = unfold_outliers(w_q, tag, T, alpha)
    # MSE.
    rtn_mse = ((x - w_rtn) ** 2).mean().item()
    dbaf_mse = ((x - x_dbaf) ** 2).mean().item()
    # Body vs tail decomposition.
    body_mask = x.abs() <= T
    tail_mask = ~body_mask
    body_mse_rtn = ((x[body_mask] - w_rtn[body_mask]) ** 2).mean().item() if body_mask.any() else 0.0
    body_mse_dbaf = ((x[body_mask] - x_dbaf[body_mask]) ** 2).mean().item() if body_mask.any() else 0.0
    tail_mse_rtn = ((x[tail_mask] - w_rtn[tail_mask]) ** 2).mean().item() if tail_mask.any() else 0.0
    tail_mse_dbaf = ((x[tail_mask] - x_dbaf[tail_mask]) ** 2).mean().item() if tail_mask.any() else 0.0
    # Bit-distinct count: how many unique quantization codes used.
    n_codes_rtn = int(torch.unique(w_rtn).numel())
    n_codes_dbaf = int(torch.unique(w_q).numel())
    return {
        "T": T,
        "frac_outlier": tail_mask.float().mean().item(),
        "rtn_mse": rtn_mse, "dbaf_mse": dbaf_mse,
        "rtn_body_mse": body_mse_rtn, "dbaf_body_mse": body_mse_dbaf,
        "rtn_tail_mse": tail_mse_rtn, "dbaf_tail_mse": tail_mse_dbaf,
        "n_codes_rtn": n_codes_rtn, "n_codes_dbaf": n_codes_dbaf,
        "gain_pct": (rtn_mse - dbaf_mse) / max(rtn_mse, 1e-12) * 100,
        "body_gain_pct": (body_mse_rtn - body_mse_dbaf) / max(body_mse_rtn, 1e-12) * 100,
        "tail_gain_pct": (tail_mse_rtn - tail_mse_dbaf) / max(tail_mse_rtn, 1e-12) * 100,
    }


def main():
    out = pathlib.Path("/home/ubuntu/unifying-ptq/results/S4-synthetic-dense-vs-sparse")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 100_000

    rows = []

    # Sparse family: vary outlier count and magnitude.
    for k in [10, 100, 1000]:  # frac = 0.01%, 0.1%, 1%
        for mag in [5.0, 8.0, 12.0]:
            x = synth_sparse(n, k, mag, rng)
            res = evaluate(x)
            res.update({"family": "sparse", "k": k, "mag": mag})
            rows.append(res)
            print(f"sparse k={k:>4} mag={mag:>5.1f}: frac={res['frac_outlier']:.4f} "
                  f"total_gain={res['gain_pct']:>+6.2f}% body_gain={res['body_gain_pct']:>+6.2f}% "
                  f"tail_gain={res['tail_gain_pct']:>+6.2f}%")

    # Dense family: vary t-distribution df.
    for df in [2.5, 3.0, 4.0, 6.0, 10.0]:
        x = synth_dense(n, df, rng)
        res = evaluate(x)
        res.update({"family": "dense", "df": df})
        rows.append(res)
        print(f"dense  df={df:>4.1f}: frac={res['frac_outlier']:.4f} "
              f"total_gain={res['gain_pct']:>+6.2f}% body_gain={res['body_gain_pct']:>+6.2f}% "
              f"tail_gain={res['tail_gain_pct']:>+6.2f}%")

    (out / "results.json").write_text(json.dumps(rows, indent=2))

    # Quick plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for fam, marker, color in [("sparse", "o", "C0"), ("dense", "s", "C3")]:
        xs = [r["frac_outlier"] for r in rows if r["family"] == fam]
        body = [r["body_gain_pct"] for r in rows if r["family"] == fam]
        tail = [r["tail_gain_pct"] for r in rows if r["family"] == fam]
        ax[0].scatter(xs, body, marker=marker, color=color, label=fam)
        ax[1].scatter(xs, tail, marker=marker, color=color, label=fam)
    ax[0].set_xlabel("frac |x|>3σ"); ax[0].set_ylabel("DBAF body-MSE gain %")
    ax[0].set_title("Body of distribution"); ax[0].axhline(0, color="k", ls=":")
    ax[0].legend(); ax[0].set_xscale("log")
    ax[1].set_xlabel("frac |x|>3σ"); ax[1].set_ylabel("DBAF tail-MSE gain %")
    ax[1].set_title("Tail (|x|>T) of distribution"); ax[1].axhline(0, color="k", ls=":")
    ax[1].legend(); ax[1].set_xscale("log")
    plt.tight_layout(); plt.savefig(out / "dense_vs_sparse.pdf"); plt.close()
    print(f"\nResults + plot in {out}")


if __name__ == "__main__":
    main()
