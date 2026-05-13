"""Evaluate FlatQuant-calibrated LLaMA-3-8B on WikiText-2 at multiple
context lengths to test the KV-cache PCSA hypothesis.

If KV-PCSA helps long-context retention, the PPL gap between
(baseline) and (KV-PCSA) should *widen* as seq_len grows.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


@torch.no_grad()
def wikitext_ppl(model, tokenizer, seq_len: int, n_chunks_cap: int = 32) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    n_chunks = min(n_chunks_cap, ids.shape[1] // seq_len)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-path", required=True,
                   help="Path to the calibration output dir (contains flat_matrices.pth)")
    p.add_argument("--kv-pcsa", action="store_true",
                   help="Set if this calibration used --kv-pcsa")
    p.add_argument("--seq-lens", nargs="+", type=int, default=[2048, 4096, 8192])
    p.add_argument("--label", required=True, help="Name for this run in results")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Construct args object compatible with FlatQuant's apply_flatquant_to_llama
    class FQArgs:
        pass
    a = FQArgs()
    a.w_bits = 4; a.a_bits = 4; a.k_bits = 4; a.v_bits = 4; a.q_bits = 16
    a.w_groupsize = -1; a.a_groupsize = -1
    a.w_asym = False; a.a_asym = False
    a.k_asym = True; a.v_asym = True
    a.k_groupsize = 128; a.v_groupsize = 128
    a.lwc = True; a.lac = True
    a.cali_trans = True; a.add_diag = True
    a.direct_inv = False
    a.kv_pcsa = args.kv_pcsa; a.kv_pcsa_anchors = 4
    a.disable_pcsa = False
    a.disable_dbaf = False
    a.separate_vtrans = False

    print(f"[eval] loading LLaMA-3-8B + reloading calibrated matrices", flush=True)
    from transformers import AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True,
    )

    from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
    apply_flatquant_to_llama(a, model)

    # Load matrices + parameters
    matrices = torch.load(f"{args.matrix_path}/flat_matrices.pth", map_location="cuda", weights_only=False)
    params = torch.load(f"{args.matrix_path}/flat_parameters.pth", map_location="cuda", weights_only=False)
    print(f"[eval] loaded {len(matrices)} matrix tensors + {len(params)} param tensors", flush=True)
    msg_m = model.load_state_dict(matrices, strict=False)
    msg_p = model.load_state_dict(params, strict=False)
    print(f"[eval] missing matrices: {len(msg_m.missing_keys)}; missing params: {len(msg_p.missing_keys)}", flush=True)

    # Optionally load PCSA state
    if pathlib.Path(f"{args.matrix_path}/pcsa_state.pth").exists():
        pcsa = torch.load(f"{args.matrix_path}/pcsa_state.pth", map_location="cuda", weights_only=False)
        model.load_state_dict(pcsa, strict=False)
        print(f"[eval] loaded PCSA state ({len(pcsa)} tensors)", flush=True)

    # Set eval mode
    for m in model.modules():
        if hasattr(m, "_eval_mode"):
            m._eval_mode = True

    model.eval()

    results = {}
    for seq_len in args.seq_lens:
        try:
            t0 = time.time()
            ppl = wikitext_ppl(model, tok, seq_len=seq_len)
            results[seq_len] = ppl
            print(f"[eval] seq_len={seq_len}: PPL={ppl:.3f}  (took {time.time()-t0:.1f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[eval] seq_len={seq_len}: OOM", flush=True)
            results[seq_len] = None
            torch.cuda.empty_cache()

    out = {"label": args.label, "kv_pcsa": args.kv_pcsa, "matrix_path": args.matrix_path,
           "wikitext2_ppl_by_seqlen": results}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
