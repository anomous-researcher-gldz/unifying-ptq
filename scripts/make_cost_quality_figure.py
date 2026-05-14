"""Cost-vs-quality figure generator for the EMNLP 2026 paper.

Reads results from:
  - PAPER_RESULTS.md (parsed for hardcoded table values)
  - Individual eval JSONs under results/ and /data/outputs/

Plots calibration cost (x-axis, log seconds) vs quality metric (y-axis):
  - LLM:   WikiText-2 PPL  (lower is better)
  - SAM:   COCO mAP        (higher is better)
  - SR:    PSNR dB         (higher is better)

Two marker classes:
  - Base methods:            'o' markers, dark colour
  - +DBAF+PCSA-tf variants:  'x' markers, lighter colour
  Each (base, +variant) pair is connected with an arrow.

Outputs:
  /home/ubuntu/paper/emnlp2026/figures/cost_quality_llm.pdf
  /home/ubuntu/paper/emnlp2026/figures/cost_quality_sam.pdf
  /home/ubuntu/paper/emnlp2026/figures/cost_quality_sr.pdf

Tolerant of missing data: if a JSON path or table cell doesn't exist, the
method is silently skipped and reported in the "missing" summary at the end.

Usage (no GPU required — reads JSON files only):
  python scripts/make_cost_quality_figure.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────
REPO = pathlib.Path("/home/ubuntu/unifying-ptq")
RESULTS = REPO / "results"
DATA_OUT = pathlib.Path("/data/outputs")
FIGURES = pathlib.Path("/home/ubuntu/paper/emnlp2026/figures")

# ──────────────────────────────────────────────────────────────────────────
# Calibration-cost estimates (seconds).
# These are rough estimates used only for x-axis placement.
# ──────────────────────────────────────────────────────────────────────────
COST_S = {
    "RTN":           1,
    "SmoothQuant":   30,
    "GPTQ":          120,
    "AWQ":           60,
    "OmniQuant":     3.6 * 3600,   # 3.6 h
    "SpinQuant":     3600,
    "FlatQuant":     7200,
    "AHCPTQ-train":  10800,
    "2DQuant":       1800,
    "CompSRT":       7200,
    # DBAF-tf variants add essentially zero calibration cost (training-free)
    "RTN+DBAF":      1,
    "GPTQ+DBAF":     120,
    "AWQ+DBAF":      60,
    "FlatQuant+DBAF+PCSA":    7200,
    "AHCPTQ+DBAF+PCSA":       10800,
    "CompSRT+DBAF":           7200,
    "2DQuant+DBAF":           1800,
}

# ── helpers ───────────────────────────────────────────────────────────────
missing_log: list[str] = []


def _load_json(path: pathlib.Path | str) -> Optional[dict]:
    """Return parsed JSON or None (logs to missing_log)."""
    p = pathlib.Path(path)
    if not p.exists():
        missing_log.append(f"missing: {p}")
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:
        missing_log.append(f"error reading {p}: {exc}")
        return None


def _get(d: dict, *keys, default=None):
    """Nested dict get with fallback."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ── data collection ───────────────────────────────────────────────────────
# Each entry: (method_name, cost_seconds, metric_value, is_dbaf_variant)
# is_dbaf_variant=True → 'x' marker + lighter colour
# is_dbaf_variant=False → 'o' marker + dark colour

def _collect_llm() -> list[tuple[str, float, float, bool]]:
    """WikiText-2 PPL for LLaMA-3-8B methods."""
    rows: list[tuple[str, float, float, bool]] = []

    def _add(name, cost_key, ppl, is_dbaf):
        if ppl is not None and not math.isnan(ppl) and not math.isinf(ppl):
            rows.append((name, COST_S[cost_key], ppl, is_dbaf))
        else:
            missing_log.append(f"LLM | {name}: ppl={ppl} (skipped)")

    # ── Hardcoded from PAPER_RESULTS.md / ICML Table 5 ───────────────────
    # FP16 baseline (reference only — no cost)
    # _add("FP16", ...) — skip; not a quantization method

    # SmoothQuant (ICML Table 5)
    _add("SmoothQuant", "SmoothQuant", 210.19, False)

    # SpinQuant (ICML Table 5)
    _add("SpinQuant", "SpinQuant", 7.96, False)

    # FlatQuant (FlatQuant README Table 1)
    _add("FlatQuant", "FlatQuant", 6.98, False)

    # FlatQuant + DBAF + PCSA (ICML Table 5)
    _add("FlatQuant+DBAF+PCSA", "FlatQuant+DBAF+PCSA", 6.96, True)

    # ── From eval JSON files ──────────────────────────────────────────────
    # RTN baseline
    d = _load_json(RESULTS / "S4-dbaf-weak/llama3-8b/rtn/baseline/eval.json")
    if d:
        _add("RTN", "RTN", _get(d, "wikitext2_ppl"), False)

    # RTN + DBAF
    d = _load_json(RESULTS / "S4-dbaf-weak/llama3-8b/rtn/with-dbaf/eval.json")
    if d:
        _add("RTN+DBAF", "RTN+DBAF", _get(d, "wikitext2_ppl"), True)

    # GPTQ baseline
    d = _load_json(RESULTS / "S4-dbaf-weak/llama3-8b/gptq/baseline/eval.json")
    if d:
        _add("GPTQ", "GPTQ", _get(d, "wikitext2_ppl"), False)

    # GPTQ + DBAF
    d = _load_json(RESULTS / "S4-dbaf-weak/llama3-8b/gptq/with-dbaf/eval.json")
    if d:
        _add("GPTQ+DBAF", "GPTQ+DBAF", _get(d, "wikitext2_ppl"), True)

    # AWQ baseline
    d = _load_json(RESULTS / "S4-dbaf-weak/llama3-8b/awq/baseline/eval.json")
    if d:
        _add("AWQ", "AWQ", _get(d, "wikitext2_ppl"), False)

    # AWQ + DBAF
    d = _load_json(RESULTS / "S4-dbaf-weak/llama3-8b/awq/with-dbaf/eval.json")
    if d:
        _add("AWQ+DBAF", "AWQ+DBAF", _get(d, "wikitext2_ppl"), True)

    # OmniQuant — look in /data/outputs/G6-omniquant-llama3-8b/
    for candidate in [
        DATA_OUT / "G6-omniquant-llama3-8b/eval.json",
        DATA_OUT / "G6-omniquant-llama3-8b/arm_A_vanilla/eval.json",
        RESULTS / "G6-omniquant-llama3-8b/eval.json",
    ]:
        d = _load_json(candidate)
        if d:
            ppl = _get(d, "wikitext2_ppl") or _get(d, "ppl") or _get(d, "wiki_ppl")
            _add("OmniQuant", "OmniQuant", ppl, False)
            break
    else:
        missing_log.append("LLM | OmniQuant: no eval.json found in G6-omniquant-llama3-8b/")

    return rows


def _collect_sam() -> list[tuple[str, float, float, bool]]:
    """COCO segm mAP for SAM-B methods."""
    rows: list[tuple[str, float, float, bool]] = []

    def _add(name, cost_key, map_val, is_dbaf):
        if map_val is not None and not math.isnan(map_val):
            rows.append((name, COST_S[cost_key], float(map_val), is_dbaf))
        else:
            missing_log.append(f"SAM | {name}: mAP={map_val} (skipped)")

    # RTN baseline (500 COCO val images)
    d = _load_json(RESULTS / "S4-dbaf-weak/sam-b-rtn/baseline/eval.json")
    if d:
        _add("RTN", "RTN", _get(d, "segm_mAP"), False)

    # RTN + DBAF
    d = _load_json(RESULTS / "S4-dbaf-weak/sam-b-rtn/with-dbaf/eval.json")
    if d:
        _add("RTN+DBAF", "RTN+DBAF", _get(d, "segm_mAP"), True)

    # AHCPTQ baseline — hardcoded from ICML Table 4 (mAP in 0-100 scale → divide by 100)
    _add("AHCPTQ-train", "AHCPTQ-train", 13.4 / 100.0, False)

    # AHCPTQ + DBAF + PCSA — ICML Table 4
    _add("AHCPTQ+DBAF+PCSA", "AHCPTQ+DBAF+PCSA", 18.2 / 100.0, True)

    # Try eval JSON paths for AHCPTQ results
    for candidate in [
        RESULTS / "S9-downstream/sam-b-ahcptq/baseline/eval.json",
        DATA_OUT / "G8-training-free-full/sam-b/baseline/eval.json",
    ]:
        d = _load_json(candidate)
        if d:
            map_v = _get(d, "segm_mAP") or _get(d, "mAP") or _get(d, "map")
            if map_v is not None:
                _add("AHCPTQ-train (json)", "AHCPTQ-train", float(map_v), False)
                break

    return rows


def _collect_sr() -> list[tuple[str, float, float, bool]]:
    """SwinIR PSNR (Set5 ×2) for SR methods."""
    rows: list[tuple[str, float, float, bool]] = []

    def _add(name, cost_key, psnr, is_dbaf):
        if psnr is not None and not math.isnan(psnr):
            rows.append((name, COST_S[cost_key], float(psnr), is_dbaf))
        else:
            missing_log.append(f"SR | {name}: PSNR={psnr} (skipped)")

    # ── From S8 eval JSONs ────────────────────────────────────────────────
    d = _load_json(RESULTS / "S8-compsrt/swinir-light-x2/baseline/eval.json")
    if d:
        psnr = _get(d, "psnr_db")
        _add("RTN", "RTN", psnr, False)

    d = _load_json(RESULTS / "S8-compsrt/swinir-light-x2/with-dbaf/eval.json")
    if d:
        psnr = _get(d, "psnr_db")
        _add("RTN+DBAF", "RTN+DBAF", psnr, True)

    # CompSRT + DBAF (Set5 ×2) — ICML Table 6 (hardcoded)
    _add("CompSRT", "CompSRT", 38.13, False)
    _add("CompSRT+DBAF", "CompSRT+DBAF", 38.15, True)

    # 2DQuant (ICML Table 6)
    _add("2DQuant", "2DQuant", 37.87, False)

    # Try to read 2DQuant eval JSON
    for candidate in [
        DATA_OUT / "G7-2dquant-swinir/eval.json",
        DATA_OUT / "G7-2dquant-swinir/x2_Set5/eval.json",
        RESULTS / "G7-2dquant-swinir/eval.json",
    ]:
        d = _load_json(candidate)
        if d:
            psnr = _get(d, "psnr_db") or _get(d, "psnr")
            if psnr:
                _add("2DQuant (json)", "2DQuant", psnr, False)
                break

    # G8 training-free full (try multiple naming conventions)
    for candidate in [
        RESULTS / "G8-training-free-full/swinir-x2-Set5/baseline/eval.json",
        RESULTS / "G8-training-free-full/eval.json",
    ]:
        d = _load_json(candidate)
        if d:
            psnr = _get(d, "psnr_db") or _get(d, "psnr")
            if psnr:
                _add("RTN (G8)", "RTN", psnr, False)
                break

    return rows


# ── plotting ──────────────────────────────────────────────────────────────
# Colour palettes: dark for base, lighter for +DBAF variant
_DARK_COLOURS = [
    "#1f4e79",  # deep blue
    "#7b2e00",  # deep orange
    "#1a5c2a",  # deep green
    "#4b0082",  # indigo
    "#5c3a00",  # dark brown
    "#005555",  # dark teal
    "#6b006b",  # dark purple
    "#3d3d00",  # dark olive
]
_LIGHT_COLOURS = [
    "#5b9bd5",  # light blue
    "#ed7d31",  # light orange
    "#70ad47",  # light green
    "#9b59b6",  # light purple
    "#c55a11",  # lighter brown
    "#00b0b0",  # light teal
    "#cc66cc",  # light purple
    "#b5b500",  # olive
]


def _build_pairs(
    rows: list[tuple[str, float, float, bool]]
) -> tuple[list, list]:
    """Separate rows into (base_rows, variant_rows), matched where possible."""
    base = [(n, c, q) for n, c, q, is_v in rows if not is_v]
    vari = [(n, c, q) for n, c, q, is_v in rows if is_v]
    return base, vari


def _method_root(name: str) -> str:
    """Strip +DBAF / +PCSA suffixes to get the base method name."""
    for suffix in ["+DBAF+PCSA", "+DBAF", " (json)", " (G8)"]:
        name = name.replace(suffix, "")
    return name.strip()


def _plot_cost_quality(
    ax: plt.Axes,
    rows: list[tuple[str, float, float, bool]],
    y_label: str,
    y_lower_is_better: bool = False,
    title: str = "",
) -> None:
    """Draw scatter plot with arrows between matched base/variant pairs."""
    base_rows, vari_rows = _build_pairs(rows)

    # Build colour map by root method
    all_roots = list(dict.fromkeys(_method_root(n) for n, *_ in rows))
    dark_map = {r: _DARK_COLOURS[i % len(_DARK_COLOURS)] for i, r in enumerate(all_roots)}
    light_map = {r: _LIGHT_COLOURS[i % len(_LIGHT_COLOURS)] for i, r in enumerate(all_roots)}

    # Build lookup for variants
    vari_by_root: dict[str, tuple[float, float]] = {
        _method_root(n): (c, q) for n, c, q in vari_rows
    }

    for name, cost, quality in base_rows:
        root = _method_root(name)
        col = dark_map[root]
        ax.scatter(cost, quality, marker="o", s=80, color=col, zorder=4, label=name)
        ax.annotate(
            name,
            (cost, quality),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6,
            color=col,
        )
        # Arrow to variant if it exists
        if root in vari_by_root:
            v_cost, v_quality = vari_by_root[root]
            ax.scatter(v_cost, v_quality, marker="x", s=80,
                       color=light_map[root], zorder=4)
            # Annotate the variant
            vname = next((n for n, c, q in vari_rows if _method_root(n) == root), "")
            ax.annotate(
                vname,
                (v_cost, v_quality),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=6,
                color=light_map[root],
            )
            ax.annotate(
                "",
                xy=(v_cost, v_quality),
                xytext=(cost, quality),
                arrowprops=dict(
                    arrowstyle="->",
                    color=col,
                    lw=1.0,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=3,
            )

    # Plot any variants that have no matching base
    plotted_roots = {_method_root(n) for n, _, _ in base_rows}
    for name, cost, quality in vari_rows:
        root = _method_root(name)
        if root not in plotted_roots:
            ax.scatter(cost, quality, marker="x", s=80,
                       color=light_map.get(root, "#888888"), zorder=4, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Calibration cost (seconds, log scale)", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=10)

    # direction hint
    hint = "↓ better" if y_lower_is_better else "↑ better"
    ax.text(
        0.98, 0.02, hint,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#555555",
    )

    # Custom legend: o = base, x = +DBAF
    base_handle = mpatches.Patch(color="#1f4e79", label="Base method (o)")
    vari_handle = mpatches.Patch(color="#5b9bd5", label="+DBAF+PCSA-tf (x)")
    ax.legend(handles=[base_handle, vari_handle], fontsize=7, loc="upper right")

    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate cost-vs-quality figures")
    p.add_argument(
        "--out-dir",
        default=str(FIGURES),
        help="Output directory for PDF figures",
    )
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── LLM figure ────────────────────────────────────────────────────────
    llm_rows = _collect_llm()
    if llm_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_cost_quality(
            ax, llm_rows,
            y_label="WikiText-2 PPL (W4, LLaMA-3-8B)",
            y_lower_is_better=True,
            title="Calibration Cost vs Quality — LLM (LLaMA-3-8B W4)",
        )
        fig.tight_layout()
        out_path = out_dir / "cost_quality_llm.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")
    else:
        print("[warn] No LLM data points collected — cost_quality_llm.pdf not written")
        missing_log.append("LLM figure: no data points")

    # ── SAM figure ────────────────────────────────────────────────────────
    sam_rows = _collect_sam()
    if sam_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_cost_quality(
            ax, sam_rows,
            y_label="COCO segm mAP (W4, SAM-B)",
            y_lower_is_better=False,
            title="Calibration Cost vs Quality — SAM-B (COCO mAP W4)",
        )
        fig.tight_layout()
        out_path = out_dir / "cost_quality_sam.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")
    else:
        print("[warn] No SAM data points — cost_quality_sam.pdf not written")
        missing_log.append("SAM figure: no data points")

    # ── SR figure ─────────────────────────────────────────────────────────
    sr_rows = _collect_sr()
    if sr_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_cost_quality(
            ax, sr_rows,
            y_label="PSNR dB (Set5 ×2, SwinIR-light W4)",
            y_lower_is_better=False,
            title="Calibration Cost vs Quality — SwinIR-light (PSNR W4)",
        )
        fig.tight_layout()
        out_path = out_dir / "cost_quality_sr.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")
    else:
        print("[warn] No SR data points — cost_quality_sr.pdf not written")
        missing_log.append("SR figure: no data points")

    # ── Missing-data report ───────────────────────────────────────────────
    if missing_log:
        print("\n[missing / skipped data cells]")
        for msg in missing_log:
            print(f"  {msg}")
    else:
        print("\n[all data cells found — no missing cells]")


if __name__ == "__main__":
    main()
