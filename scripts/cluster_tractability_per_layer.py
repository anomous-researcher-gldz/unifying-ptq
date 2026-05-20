"""GAP-1: Per-layer cluster tractability — proves SAM > LLM gap isn't site-cherry-picked.

The published 0.189 (SAM-B mask decoder) vs 0.790 (LLaMA-3-8B q_proj) numbers
aggregate descriptors across layers. A reviewer can ask: did you pick the
worst LLM layer and best SAM layer? Here we measure compactness at EACH layer
independently for both models and report min/max/median.

If LLaMA's BEST per-layer compactness is still worse than SAM's WORST,
the gap is structural, not site-dependent.
"""
from __future__ import annotations
import sys, json, pathlib, argparse, gc, warnings
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")
warnings.filterwarnings("ignore")


def _kmeans(X, k, iters=50, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=1, max_iter=iters, random_state=seed).fit(X)
    a = km.labels_
    C = km.cluster_centers_
    d_self = np.linalg.norm(X - C[a], axis=1)
    return a, C, d_self


def _baseline_permute(X, seed):
    rng = np.random.default_rng(seed + 7)
    Xp = X.copy()
    for j in range(X.shape[1]):
        Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp


def compactness_for(X, k, n_baseline=10):
    X = X.astype(np.float64)
    _, _, d_real = _kmeans(X, k)
    ratios = []
    for s in range(n_baseline):
        _, _, d_perm = _kmeans(_baseline_permute(X, s), k, seed=s)
        ratios.append(d_real.mean() / max(d_perm.mean(), 1e-9))
    return float(np.mean(ratios)), float(np.std(ratios))


@torch.no_grad()
def collect_sam_per_layer(n_images=50):
    import segment_anything as sa
    sam = sa.sam_model_registry["vit_b"](
        checkpoint="/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth"
    ).cuda().eval()
    md = sam.mask_decoder
    n_blocks = len(md.transformer.layers)
    per_layer = {i: [] for i in range(n_blocks)}

    def make_hook(idx):
        def hook(mod, args):
            q = args[0]
            d = torch.nn.functional.normalize(q.float().mean(dim=1), dim=-1)
            per_layer[idx].append(d.cpu().numpy())
        return hook

    handles = []
    for i, blk in enumerate(md.transformer.layers):
        handles.append(blk.cross_attn_token_to_image.q_proj.register_forward_pre_hook(make_hook(i)))

    torch.manual_seed(0)
    for i in range(n_images):
        s = 30.0 + 6.0 * i
        img = (torch.randn(1, 3, 1024, 1024) * s + 128.0).clamp(0, 255).cuda() / 255.0
        img_emb = sam.image_encoder(img)
        n_pts = 2 + (i % 3)
        coords = torch.rand(1, n_pts, 2, device="cuda") * 1024.0
        labels = torch.ones(1, n_pts, device="cuda")
        sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        _ = md(image_embeddings=img_emb,
               image_pe=sam.prompt_encoder.get_dense_pe(),
               sparse_prompt_embeddings=sparse,
               dense_prompt_embeddings=dense, multimask_output=False)
    for h in handles: h.remove()
    return {i: np.concatenate(per_layer[i], axis=0) for i in range(n_blocks)}


@torch.no_grad()
def collect_llama_per_layer(n_prompts=50):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
    ).cuda().eval()
    n_layers = len(model.model.layers)
    per_layer = {i: [] for i in range(n_layers)}

    def make_hook(idx):
        def hook(mod, args):
            x = args[0]
            d = torch.nn.functional.normalize(x.float().mean(dim=1), dim=-1)
            per_layer[idx].append(d.cpu().numpy())
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook(i)))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 300][:n_prompts]
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        _ = model(ids)
    for h in handles: h.remove()
    return {i: np.concatenate(per_layer[i], axis=0) for i in range(n_layers)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_inputs", type=int, default=50)
    p.add_argument("--k_sam", type=int, default=4)
    p.add_argument("--k_llm", type=int, default=8)
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/cluster_tractability_per_layer.json")
    args = p.parse_args()

    print("=== SAM-B per-layer ===", flush=True)
    sam_lay = collect_sam_per_layer(args.n_inputs)
    sam_rows = []
    for i, X in sam_lay.items():
        c, sd = compactness_for(X, args.k_sam)
        sam_rows.append({"layer": i, "n": int(X.shape[0]), "d": int(X.shape[1]),
                         "compactness": c, "std": sd})
        print(f"  SAM layer {i}: n={X.shape[0]} d={X.shape[1]} c={c:.4f} ± {sd:.4f}", flush=True)
    del sam_lay; gc.collect(); torch.cuda.empty_cache()

    print("\n=== LLaMA-3-8B per-layer ===", flush=True)
    llm_lay = collect_llama_per_layer(args.n_inputs)
    llm_rows = []
    for i, X in llm_lay.items():
        c, sd = compactness_for(X, args.k_llm)
        llm_rows.append({"layer": i, "n": int(X.shape[0]), "d": int(X.shape[1]),
                         "compactness": c, "std": sd})
        if i < 4 or i >= len(llm_lay) - 2 or i % 4 == 0:
            print(f"  LLM layer {i}: c={c:.4f} ± {sd:.4f}", flush=True)

    sam_c = [r["compactness"] for r in sam_rows]
    llm_c = [r["compactness"] for r in llm_rows]
    summary = {
        "sam_min": float(np.min(sam_c)), "sam_max": float(np.max(sam_c)),
        "sam_median": float(np.median(sam_c)), "sam_n_layers": len(sam_c),
        "llm_min": float(np.min(llm_c)), "llm_max": float(np.max(llm_c)),
        "llm_median": float(np.median(llm_c)), "llm_n_layers": len(llm_c),
        "k_sam": args.k_sam, "k_llm": args.k_llm,
        "gap_question": "Is SAM's worst (highest compactness) < LLM's best (lowest)?",
        "sam_worst": float(np.max(sam_c)), "llm_best": float(np.min(llm_c)),
        "structural_gap": float(np.max(sam_c)) < float(np.min(llm_c)),
    }
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  SAM-B (K={args.k_sam}): min={summary['sam_min']:.4f} max={summary['sam_max']:.4f} median={summary['sam_median']:.4f} n_layers={summary['sam_n_layers']}", flush=True)
    print(f"  LLaMA-3-8B (K={args.k_llm}): min={summary['llm_min']:.4f} max={summary['llm_max']:.4f} median={summary['llm_median']:.4f} n_layers={summary['llm_n_layers']}", flush=True)
    print(f"  SAM worst ({summary['sam_worst']:.4f}) < LLM best ({summary['llm_best']:.4f}) ? {summary['structural_gap']}", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({
        "summary": summary, "sam_layers": sam_rows, "llm_layers": llm_rows,
    }, indent=2))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
