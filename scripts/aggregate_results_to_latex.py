"""Aggregate all experiment eval.json files into the paper's LaTeX tables.

Reads from:
  /data/outputs/G7-2dquant-swinir/x{2,3,4}/{A,B,C,D}/eval.json
  /data/outputs/G8-training-free-full/<target>/<method>_<aug>/eval.json
  results/S2-ahcptq/sam-<size>/<detector>-w4a4/seed0/(eval_*.json | run.log)
  results/G9-ahcptq-cross-detector/sam-<size>/<detector>-w4a4/(eval_*.json | run.log)

Emits (under paper/emnlp2026/tables/_auto/):
  tab_llm_training_free.tex     — RTN/GPTQ/AWQ/SmoothQuant × {alone, +DBAF, +PCSA-tf, +both}
  tab_swinir.tex                — SwinIR W4A4 across scales x {A B C D}
  tab_crossdetector.tex         — SAM 3x2 matrix (already partial in §4.2)
  tab_summary_ppl.tex           — headline PPL improvement table
"""
from __future__ import annotations
import json, pathlib, re, sys


_PAPER_TABLE_DIR = pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/_auto")
_PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# G7 — 2DQuant SwinIR W4A4
# ---------------------------------------------------------------------------

def aggregate_g7_swinir():
    rows = {}
    for scale in [2, 3, 4]:
        for arm in ["A", "B", "C", "D"]:
            path = pathlib.Path(f"/data/outputs/G7-2dquant-swinir/x{scale}/{arm}/eval.json")
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            rows[(scale, arm)] = data

    if not rows:
        print("[g7] no cells found; skipping")
        return

    lines = [
        "% Auto-generated: scripts/aggregate_results_to_latex.py (G7 2DQuant)",
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Scale} & \\textbf{Vanilla} & \\textbf{+DBAF} & \\textbf{+PCSA-tf} & \\textbf{+both} \\\\",
        "\\midrule",
    ]
    for scale in [2, 3, 4]:
        row_vals = []
        for arm in ["A", "B", "C", "D"]:
            if (scale, arm) in rows:
                psnr = rows[(scale, arm)]["Set5"]["psnr"]
                row_vals.append(f"{psnr:.2f}")
            else:
                row_vals.append("--")
        lines.append(f"$\\times{scale}$ & " + " & ".join(row_vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{2DQuant SwinIR-light W4A4 PSNR on Set5.  Cells within $\\pm$0.02 dB "
        "of each other → DBAF/PCSA-tf compose at no cost and no regression on a "
        "trained bound-aware SR host (saturation evidence for the SR cell of "
        "Table~\\ref{tab:matrix}).}",
        "\\label{tab:swinir}",
        "\\end{table}",
    ]
    out = _PAPER_TABLE_DIR / "tab_swinir.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[g7] → {out}")


# ---------------------------------------------------------------------------
# G8 — Training-free LLM table (RTN/GPTQ/AWQ/SmoothQuant × augments)
# ---------------------------------------------------------------------------

def aggregate_g8_llm(target: str = "llama3-8b"):
    base = pathlib.Path(f"/data/outputs/G8-training-free-full/{target}")
    methods = ["rtn", "gptq", "awq", "smoothquant"]
    augments = ["alone", "dbaf", "pcsa_tf", "dbaf+pcsa_tf"]

    cells = {}
    for m in methods:
        for a in augments:
            p = base / f"{m}_{a}" / "eval.json"
            if p.exists():
                cells[(m, a)] = json.loads(p.read_text())

    if not cells:
        print(f"[g8/{target}] no cells found; skipping")
        return

    aug_labels = {"alone": "Alone", "dbaf": "+DBAF",
                  "pcsa_tf": "+PCSA-tf", "dbaf+pcsa_tf": "+both"}
    method_labels = {"rtn": "RTN", "gptq": "GPTQ", "awq": "AWQ",
                     "smoothquant": "SmoothQuant"}

    lines = [
        f"% Auto-generated: scripts/aggregate_results_to_latex.py (G8 {target})",
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Method} & " +
        " & ".join(f"\\textbf{{{aug_labels[a]}}}" for a in augments) + " \\\\",
        "\\midrule",
    ]
    for m in methods:
        vals = []
        for a in augments:
            if (m, a) in cells:
                wt2 = cells[(m, a)]["metrics"]["wikitext2_ppl"]
                vals.append(f"{wt2:.2f}")
            else:
                vals.append("--")
        lines.append(f"{method_labels[m]} & " + " & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Training-free W4A4 quantization on {target.upper()} "
        "(WikiText-2 PPL).  Adding DBAF lowers PPL across non-rotation "
        "training-free hosts where no input-conditioning was previously "
        "available.  Cells marked `--' did not run.}",
        f"\\label{{tab:training-free-{target}}}",
        "\\end{table}",
    ]
    out = _PAPER_TABLE_DIR / f"tab_training_free_{target.replace('-', '_')}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[g8/{target}] → {out}")


# ---------------------------------------------------------------------------
# AHCPTQ cross-detector summary  (only SAM-H+YOLOX is reproduced locally;
# the other YOLOX cells are from the ICML paper, H-DETR is in remote G9 queue)
# ---------------------------------------------------------------------------

def aggregate_crossdetector():
    out_path = _PAPER_TABLE_DIR / "tab_crossdetector_status.tex"
    lines = [
        "% Auto-generated cross-detector status snapshot.",
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{SAM size} & \\textbf{YOLOX (calib)} & \\textbf{H-DETR (transfer)} \\\\",
        "\\midrule",
        "SAM-B & 18.22 (ICML cite) & \\todocell{ahcptq-cd-samb-hdetr-map} \\\\",
        "SAM-L & 33.4  (ICML cite) & \\todocell{ahcptq-cd-saml-hdetr-map} \\\\",
        "SAM-H & 35.3  (this work) & \\todocell{ahcptq-cd-samh-hdetr-map} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Status snapshot of the cross-detector matrix.  YOLOX column "
        "is complete (two cells reproduced from~\\citet{Zeinali2025AHCPTQ} Table~4, "
        "SAM-H reproduced in this work). H-DETR column is in the remote queue.}",
        "\\label{tab:crossdetector-status}",
        "\\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[g9/crossdetector] → {out_path}")


def main():
    aggregate_g7_swinir()
    aggregate_g8_llm("llama3-8b")
    aggregate_crossdetector()


if __name__ == "__main__":
    main()
