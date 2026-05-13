"""Run experiment A: DBAF on weak baselines (RTN/GPTQ/AWQ) for one (model, baseline, with-DBAF?) cell.

Outputs WikiText-2 perplexity per cell as JSON to --out.
"""
import argparse, json, pathlib, sys, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")

MODELS = {
    "llama3-8b": "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
    "qwen25-7b": "/data/modelzoo/Qwen/Qwen2.5-7B",
}
BASELINES = ["rtn", "gptq", "awq"]


def get_baseline(name):
    if name == "rtn":
        from flatquant.baselines.rtn import quantize_model
    elif name == "gptq":
        from flatquant.baselines.gptq import quantize_model
    elif name == "awq":
        from flatquant.baselines.awq import quantize_model
    return quantize_model


@torch.no_grad()
def wikitext_ppl(model: nn.Module, tokenizer, seq_len: int = 2048, n_samples: int = 64) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    n_chunks = min(n_samples, ids.shape[1] // seq_len)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


def calib_batch(tokenizer, seq_len: int = 2048, n: int = 4):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n * seq_len].view(n, seq_len).cuda()
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--baseline", choices=BASELINES, required=True)
    p.add_argument("--use-dbaf", action="store_true")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--cali-bsz", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--ppl-samples", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[S4] loading {args.model} from {MODELS[args.model]}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model],
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    calib = calib_batch(tok, seq_len=args.seq_len, n=args.cali_bsz)

    quantize_model = get_baseline(args.baseline)
    t0 = time.time()
    if args.baseline == "rtn":
        model = quantize_model(model, bits=args.bits, use_dbaf=args.use_dbaf)
    else:
        model = quantize_model(model, bits=args.bits, calibration_data=calib, use_dbaf=args.use_dbaf)
    t_quant = time.time() - t0
    print(f"[S4] quantize took {t_quant:.1f}s", flush=True)

    ppl = wikitext_ppl(model, tok, seq_len=args.seq_len, n_samples=args.ppl_samples)
    out = {
        "model": args.model,
        "baseline": args.baseline,
        "use_dbaf": args.use_dbaf,
        "bits": args.bits,
        "wikitext2_ppl": ppl,
        "quant_seconds": t_quant,
    }
    p_out = pathlib.Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
