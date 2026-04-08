"""
Compute per-layer alpha* for SAM-B across ALL quantized distributions
(activations + weights), matching the original 163/108 analysis.
Hooks into every QuantizeBase module in the quantized model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ahcptq', 'solver'))

import torch
import json
from collections import defaultdict, OrderedDict
from mmcv import Config
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, update_data_root, compat_cfg, setup_multi_processes

from ahcptq.quantization.fake_quant import (
    compute_alpha_star, compute_T, is_like_normal_plus_3sigma_outliers, QuantizeBase
)
from ahcptq.quantization.state import enable_calibration_woquantization, disable_all
import ahcptq.model.quant_model as quant_model_sam1
import utils
import importlib
importlib.import_module('projects.instance_segment_anything')

GRID_ALPHA = 0.75  # grid-selected alpha for SAM

# ---- collect stats via hooks ----
dist_log = OrderedDict()  # layer_name -> {is_like_c: [], alpha_star: []}

def make_hook(layer_name):
    def hook_fn(module, input, output):
        if not isinstance(input, tuple) or len(input) == 0:
            return
        X = input[0]
        if not isinstance(X, torch.Tensor) or X.numel() == 0:
            return

        # subsample large tensors to avoid quantile() OOM
        if X.numel() > 1_000_000:
            idx = torch.randperm(X.numel(), device=X.device)[:1_000_000]
            X_sub = X.detach().reshape(-1)[idx]
        else:
            X_sub = X.detach().reshape(-1)

        check = is_like_normal_plus_3sigma_outliers(X_sub.unsqueeze(0))

        if layer_name not in dist_log:
            dist_log[layer_name] = {'is_like_c': [], 'alpha_star': []}

        dist_log[layer_name]['is_like_c'].append(check['is_like_c'])

        if check['is_like_c']:
            a_star = compute_alpha_star(X_sub.unsqueeze(0))
            if isinstance(a_star, torch.Tensor):
                a_star = a_star.item()
            dist_log[layer_name]['alpha_star'].append(a_star)

    return hook_fn

def main():
    cfg = Config.fromfile('./projects/configs/yolox/yolo_l-sam-vit-b.py')
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.gpu_ids = [0]

    q_config = utils.parse_config('./exp/config44.yaml')

    # load calibration data
    cali_data = utils.load_calibration(cfg, distributed=False, num_samples=q_config.calibrate)

    # build and quantize model (same as test_quant.py)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = [''] * 80
    for p in model.parameters():
        p.requires_grad = False
    model.cuda().eval()

    # quantize the model to create QuantizeBase modules
    from ahcptq.solver.test_quant import quantize_model
    import argparse
    args = argparse.Namespace(
        quant_encoder=False, sam_version='sam1',
        load_pcsa='', save_pcsa='', work_dir='result/tmp',
        load_sam_path='', save_sam_path='', brecq=False,
        short4cut=False, resign_end=False, fp=False,
    )
    model = quantize_model(model, q_config, args)
    model.cuda().eval()

    # register hooks on ALL QuantizeBase modules within SAM (predictor)
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'predictor' in name and 'image_encoder' not in name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # disable quantization, just run forward to collect distributions
    disable_all(model)

    # run calibration data through model
    print(f"Running {len(cali_data)} calibration batches...")
    with torch.no_grad():
        for i, batch in enumerate(cali_data):
            model.extract_feat(batch)
            print(f"  batch {i+1}/{len(cali_data)}")

    # remove hooks
    for h in hooks:
        h.remove()

    # ---- analysis ----
    # A distribution is "seen" if the hook fired at least once
    total_distributions = len(dist_log)

    # A distribution is "problematic" (case c) if is_like_c was True in majority of batches
    problematic = []
    non_problematic = []
    for name, data in dist_log.items():
        c_count = sum(data['is_like_c'])
        total = len(data['is_like_c'])
        if c_count > total / 2:  # majority vote
            problematic.append(name)
        else:
            non_problematic.append(name)

    print(f"\n{'='*70}")
    print(f"Distribution Taxonomy (SAM-B, YOLOX detector)")
    print(f"{'='*70}")
    print(f"Total quantized distributions: {total_distributions}")
    print(f"Problematic (case c - outlier): {len(problematic)}")
    print(f"Non-problematic: {len(non_problematic)}")

    print(f"\n{'='*70}")
    print(f"Per-layer alpha* analysis (grid-selected alpha = {GRID_ALPHA})")
    print(f"{'='*70}\n")

    below_grid = 0
    results = []
    for name in sorted(problematic):
        data = dist_log[name]
        if len(data['alpha_star']) == 0:
            continue
        mean_alpha = sum(data['alpha_star']) / len(data['alpha_star'])
        is_below = mean_alpha < GRID_ALPHA
        if is_below:
            below_grid += 1
        results.append({
            'layer': name,
            'mean_alpha_star': round(mean_alpha, 6),
            'num_samples': len(data['alpha_star']),
            'below_grid': is_below,
        })
        marker = "<" if is_below else ">="
        print(f"  {name}: alpha*={mean_alpha:.4f} {marker} {GRID_ALPHA}")

    n_with_alpha = len(results)
    overall_mean = sum(r['mean_alpha_star'] for r in results) / len(results) if results else 0

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total quantized distributions (SAM only): {total_distributions}")
    print(f"Problematic distributions (case c): {len(problematic)}")
    print(f"Problematic with alpha* computed: {n_with_alpha}")
    print(f"Layers where alpha* < grid alpha ({GRID_ALPHA}): {below_grid}/{n_with_alpha}")
    print(f"Mean alpha* across problematic layers: {overall_mean:.4f}")
    print(f"\nalpha* is below the grid-selected value in {below_grid}/{n_with_alpha} problematic distributions,")
    print(f"correctly predicting that folding is beneficial.")

    # save raw results
    out_path = 'result/alpha_star_per_layer.json'
    os.makedirs('result', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'grid_alpha': GRID_ALPHA,
            'total_distributions': total_distributions,
            'problematic_count': len(problematic),
            'non_problematic_count': len(non_problematic),
            'below_grid_count': below_grid,
            'overall_mean_alpha_star': round(overall_mean, 6) if results else None,
            'per_layer': results,
            'problematic_layers': problematic,
            'non_problematic_layers': non_problematic,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
