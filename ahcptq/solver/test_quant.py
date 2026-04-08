# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import pandas as pd
from py import log
from sympy import im
import torch
from mmcv import Config, DictAction
from mmcv.utils import get_logger
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from ahcptq.quantization.quantized_module import (
    QuantizedModule,
    Quantizer,
    PreQuantizedLayer,
    QuantizedLayer,
)
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import logging
import torch.nn as nn
import torch.distributed as dist
import ahcptq.model.quant_model as quant_model_sam1
import ahcptq.model.quant_model_sam2 as quant_model_sam2
from ahcptq.model.quant_model_sam2 import QuantSAM2MLP,QuantSAM2Attention
from ahcptq.model.quant_model import QuantDecoderOurAttentionBlock
from ahcptq.model.prompt_anchor import PromptAnchorBank
from ahcptq.quantization.fake_quant import AnchorAwareFakeQuantize
# These will be assigned in main()
specials = quant_model_sam1.specials
bimodal_adjust = quant_model_sam1.bimodal_adjust
from ahcptq.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from ahcptq.quantization.quantized_module import QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul
from ahcptq.quantization.fake_quant import QuantizeBase
from ahcptq.quantization.fake_quant_blocks import BlockQuantizeBase
from ahcptq.quantization.observer import ObserverBase, AvgMSEObserver, AvgMinMaxObserver
from recon import reconstruction
import utils

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class QuantSpec:
    w_bits: int = 4
    per_channel: bool = False       # <-- per-tensor weights
    symmetric: bool = False         # if True: no zero-point
    scale_dtype_bytes: int = 2      # fp16 scale
    zp_dtype_bytes: int = 1         # uint8 zp (0 if symmetric)
    include_bias_fp32: bool = True
    include_buffers: bool = True

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def estimate_effective_model_size_mb(
    model: nn.Module,
    q: QuantSpec = QuantSpec(),
) -> Tuple[float, Dict[str, float]]:
    weight_bytes = 0
    meta_bytes = 0
    bias_bytes = 0
    other_param_bytes = 0
    buffer_bytes = 0

    quantized_param_ids = set()
    bias_param_ids = set()

    # Conv/Linear weights: W-bit packed + metadata (per-tensor => 1 scale / 1 zp per layer)
    for m in model.modules():
        if not isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue

        if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor) and (m.weight is not None):
            w = m.weight
            weight_bytes += _ceil_div(w.numel() * q.w_bits, 8)
            quantized_param_ids.add(id(m.weight))

            # metadata
            n_scales = len(q.scale)
            n_zp = 0 if q.symmetric else 1
            meta_bytes += n_scales * q.scale_dtype_bytes
            meta_bytes += n_zp * q.zp_dtype_bytes

        if q.include_bias_fp32 and hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and (m.bias is not None):
            b = m.bias
            bias_bytes += b.numel() * b.element_size()
            bias_param_ids.add(id(m.bias))

    # Everything else (non Conv/Linear weights, norms, embeddings, etc.) at true dtype size
    for p in model.parameters():
        pid = id(p)
        if pid in quantized_param_ids or pid in bias_param_ids:
            continue
        other_param_bytes += p.numel() * p.element_size()

    # Buffers (running stats, registered tensors)
    if q.include_buffers:
        for b in model.buffers():
            buffer_bytes += b.numel() * b.element_size()

    total_bytes = weight_bytes + meta_bytes + bias_bytes + other_param_bytes + buffer_bytes
    mb = total_bytes / (1024 ** 2)

    breakdown = {
        "quant_weights_MB": weight_bytes / (1024 ** 2),
        "quant_metadata_MB": meta_bytes / (1024 ** 2),
        "bias_MB": bias_bytes / (1024 ** 2),
        "other_params_MB": other_param_bytes / (1024 ** 2),
        "buffers_MB": buffer_bytes / (1024 ** 2),
        "total_MB": mb,
    }
    return mb, breakdown

def model_size_mb(model: nn.Module, *, include_buffers: bool = True) -> float:
    """
    Size in MB of all tensors that will be saved in a checkpoint:
    - parameters (weights, learnable scales, etc.)
    - buffers (observer stats, running min/max, activation quant scales if buffers, etc.)
    """
    total_bytes = 0

    # Parameters (includes weight quant scales if nn.Parameter)
    for p in model.parameters(recurse=True):
        if p is None:
            continue
        total_bytes += p.numel() * p.element_size()

    # Buffers (includes activation quantizer buffers like scale/zero_point if register_buffer)
    if include_buffers:
        for b in model.buffers(recurse=True):
            if b is None:
                continue
            total_bytes += b.numel() * b.element_size()

    return total_bytes / (1024 ** 2)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--sam_version',
        default='sam1',
        choices=['sam1', 'sam2'],
        help='Which Segment Anything variant is used in the detector (sam1 or sam2).'
    )
    parser.add_argument('--config',
                        default='./projects/configs/yolox/yolo_l-sam-vit-b.py', 
                        help='test config file path')
    parser.add_argument(
        '--work-dir',
        default='result/tmp',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='segm',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    
    parser.add_argument(
        '--quant-encoder',
        action='store_true',
        help='whether to quant encoder.')
    
    parser.add_argument(
        '--fp',
        action='store_true',
        help='fp')

    parser.add_argument(
        '--save_sam_path',
        type=str,
        default='',
        help='save_sam_path')

    parser.add_argument(
        '--load_sam_path',
        type=str,
        default='',
        help='load_sam_path')
    
    parser.add_argument(
        '--load-pcsa',
        type=str,
        default='',
        help='Path to a PCSA checkpoint to load (skips PCSA calibration)')

    parser.add_argument(
        '--save-pcsa',
        type=str,
        default='',
        help='Path to save PCSA checkpoint after calibration')

    parser.add_argument(
        '--short4cut',
        action='store_true',
        help='short cut for G mem')
    
    parser.add_argument(
        '--brecq',
        action='store_true',
        help='Brecq')
    
    parser.add_argument(
        '--resign_end',
        action='store_true',
        help='resign_end')

    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--q_config',
        type=str,
        default='./exp/config66.yaml',
        help='quantization config files')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    global logger, specials, bimodal_adjust
    args = parse_args()
    brecq = args.brecq
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    # print(args.sam_version)
    if args.sam_version == 'sam2':
        specials = quant_model_sam2.specials
        # print(specials)
        bimodal_adjust = quant_model_sam2.bimodal_adjust
    else:
        specials = quant_model_sam1.specials
        bimodal_adjust = quant_model_sam1.bimodal_adjust

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # print(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif (cfg.model.get('backbone', None) is not None
          and 'init_cfg' in cfg.model.backbone):
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    if args.q_config:
        q_config = utils.parse_config(args.q_config)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(args.work_dir, f'{timestamp}.log')
        logger = get_logger(name='ahcptq', log_file=log_file, log_level=logging.INFO)
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    cali_data = utils.load_calibration(cfg, distributed, q_config.calibrate)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # print(cfg.get('test_cfg'))
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # print(model)
    # print(model.predictor.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = {}
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if not args.fp:
        if len(args.load_sam_path) == 0:
            logger.info('quant encoder')
            logger.info(args.quant_encoder)
            logger.info('cag')
            logger.info(q_config.ahcptq.cag)
            logger.info('hluq')
            logger.info(q_config.ahcptq.hluq)
            logger.info('do sign')
            logger.info(q_config.ptq4sam.BIG)
            logger.info('do log_quant')
            logger.info(q_config.ptq4sam.AGQ)

            assert args.q_config is not None
            for para in model.parameters():
                para.requires_grad = False
            model = model.to("cuda")
            # print(model.predictor.device)
            # from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
            # from torchao.utils import unwrap_tensor_subclass

            # sam = model.predictor.model  # or wherever your SAM/SAM2 nn.Module lives

            # sam.eval().cuda()

            # # Apply TorchAO dynamic int8 quantization (real kernels, not fake-quant)
            # quantize_(sam, Int8DynamicActivationInt8WeightConfig())  # A8W8

            model = quantize_model(model, q_config, args)
            model.cuda()
            # # print(model.predictor.device)
            model.eval()
            fp_model = copy.deepcopy(model)
            fp_model = fp_model.to("cuda") 
            # # print(fp_model.predictor.device)
            # disable_all(model) 
            # enable_quantization(model)

            # spec = QuantSpec(
            # w_bits=4,
            # per_channel=False,   # per-tensor
            # symmetric=False,     # if your weights are symmetric set True
            # scale_dtype_bytes=4, # fp32
            # zp_dtype_bytes=4,    # fp32
            # )

            # total_mb, breakdown = estimate_effective_model_size_mb(model, spec)
            # print(f"Effective model size: {model_size_mb(model):.2f} MB")
            # print(breakdown)
            # disable_all(fp_model)
            pcsa_loaded = False
            if args.load_pcsa:
                load_pcsa_checkpoint(model, args.load_pcsa)
                pcsa_loaded = True

            calibrate(model, cali_data, q_config.ptq4sam.BIG, pcsa_loaded=pcsa_loaded)

            if not pcsa_loaded:
                save_path = args.save_pcsa if args.save_pcsa else os.path.join(args.work_dir, 'pcsa_checkpoint.pt')
                save_pcsa_checkpoint(model, save_path)

            if hasattr(q_config, 'recon'):
                if rank == 0:
                    logger.info('begin to do reconstruction')
                recon_model(model, fp_model, cali_data, q_config.recon, q_config.ahcptq, brecq=brecq,sam_version=args.sam_version)
           
            calculate_bitwidths(model)
            enable_quantization(model)

            for n, m in model.named_modules():
                if hasattr(m, 'drop_prob'):
                    m.drop_prob = 1
            
            if len(args.save_sam_path) > 0:
                try:
                    save_path = args.save_sam_path
                    logger.info('save sam:')
                    logger.info(save_path)
                    torch.save(model.predictor, save_path)
                except:
                    logger.info('err no save')
        else:
            load_path = args.load_sam_path
            logger.info('load sam:')
            logger.info(load_path)
            model.predictor = torch.load(load_path)
    
    model.det_model.cuda()
    
    if not distributed:
        if args.show_dir is not None and 'gt' in args.show_dir:
            gt = True
        else:
            gt = False
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr, gt=gt)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        # In multi_gpu_test, if tmpdir is None, some tesnors
        # will init on cuda by default, and no device choice supported.
        # Init a tmpdir to avoid error on npu here.
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            logger.info(q_config)
            logger.info(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

def quantize_model(model, config_quant, args):
    # print("here")
    # , config_quant.ahcptq
    # import inspect

    # however you currently build the model in test_quant.py
    # do that here and stop right after "model" is created
    #our SAM2 detector wrapper (DetWrapperInstanceSAM2 or similar)


    def replace_module(module, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config, qoutput=True):
        if isinstance(module, (QuantizedModule, PreQuantizedLayer, QuantizedLayer)):
            return
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (QuantizedModule, PreQuantizedLayer, QuantizedLayer)):
                # print("aready quant", name)
                continue
            if 'patch_embed' in name or 'output_upscaling' in name or 'iou_prediction_head' in name or 'output_hypernetworks_mlps' in name:
                continue
            if type(child_module) in specials:
                # print("special", name)
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                # print("conv/linear", name)
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.GELU)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, nn.Identity())
                else:
                    pass
            elif isinstance(child_module, nn.Identity):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config, qoutput)
    # if args.quant_encoder:
    #     model.predictor.model.image_encoder = specials[type(model.predictor.model.image_encoder)](model.predictor.model.image_encoder, config_quant.w_qconfig, config_quant.a_qconfig, config_quant.ahcptq, config_quant.ptq4sam)
    # for name, module in model.named_children():
    #     print(f"  {name}: {type(module)}")
    if args.quant_encoder:
        if args.sam_version == 'sam2':
            # print("here")
            # Use the generic SAM2 encoder wrapper we just defined
            model.predictor.model.image_encoder = specials[
                type(model.predictor.model.image_encoder)
            ](
                model.predictor.model.image_encoder,
                config_quant.w_qconfig,
                config_quant.a_qconfig,
                config_quant.ahcptq,
                config_quant.ptq4sam,
            )
            # print(model.predictor.model.image_encoder)
        else:
            # Original SAM1 behavior: use specials dict
            model.predictor.model.image_encoder = specials[
                type(model.predictor.model.image_encoder)
            ](
                model.predictor.model.image_encoder,
                config_quant.w_qconfig,
                config_quant.a_qconfig,
                config_quant.ahcptq,
                config_quant.ptq4sam,
            )

      # --- decoder / transformer quant ---
    if args.sam_version == 'sam2':
        # SAM2Base has no .mask_decoder; just walk the whole SAM2 model.
        # replace_module will only wrap modules whose type is in `specials`
        # (SAM2Attention, SAM2TwoWayAttentionBlock), so this is safe.
        
        replace_module(
            model.predictor.model,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
            config_quant.ahcptq,
            config_quant.ptq4sam,
        )
        # print(model.predictor.model)
    else:
        # SAM1 behavior: only walk mask_decoder subtree
        replace_module(
            model.predictor.model.mask_decoder,
            config_quant.w_qconfig,
            config_quant.a_qconfig,
            config_quant.ahcptq,
            config_quant.ptq4sam,
        )

    'set first layer\'s weight to 8-bit'
    # w_list, a_list = [], []
    # for name, module in model.named_modules():
    #     if isinstance(module, QuantizeBase) and 'weight' in name:
    #         # print(module)
    #         w_list.append(module)
    #     if isinstance(module, QuantizeBase) and 'act' in name:
    #         a_list.append(module)
    # w_list[0].set_bit(32)
    # a_list[0].set_bit(32)
    # print(model)
    # model = model.to("cuda")
    # print("=== Quant Model Parameters ===")
    # for name, p in model.predictor.named_parameters():
    #     print(f"{name:60s}  requires_grad={p.requires_grad}")
    # print(model)
    # print(model.predictor.model)
    return model

@torch.no_grad()
def calculate_bitwidths(model):
    count  = 0
    summed = 0.0
    if hasattr(model, "predictor") and hasattr(model.predictor, "model"):
        model = model.predictor.model  
    
    for name, module in model.named_modules():
        # print(module)
        if (isinstance(module, QuantizeBase) and 'weight' in name):
            # print(name,module.bit)
            count += 1
            summed += module.bit 
           
        if isinstance(module, QuantizeBase) and 'act' in name:
            # print(name,module.bit)
            count += 1
            summed += module.bit 
            
    # w_list[0].set_bit(32)
    # a_list[0].set_bit(32)
    avg = summed/count 
    print("average bitwidth", avg, flush=True ) 
def save_pcsa_checkpoint(model, path):
    """Save all PromptAnchorBank and AnchorAwareFakeQuantize state keyed by module path."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, PromptAnchorBank):
            state[name] = {'type': 'PromptAnchorBank', 'state_dict': module.state_dict()}
        elif isinstance(module, AnchorAwareFakeQuantize):
            state[name] = {
                'type': 'AnchorAwareFakeQuantize',
                'scale': module.scale.data.clone().cpu(),
                'zero_point': module.zero_point.clone().cpu(),
                'num_anchors': module.num_anchors,
            }
    torch.save(state, path)
    logger.info('Saved PCSA checkpoint to {} ({} modules)'.format(path, len(state)))


def load_pcsa_checkpoint(model, path):
    """Load PromptAnchorBank and AnchorAwareFakeQuantize state from a PCSA checkpoint."""
    state = torch.load(path, map_location='cpu')
    loaded = 0
    for name, module in model.named_modules():
        if name not in state:
            continue
        entry = state[name]
        if isinstance(module, PromptAnchorBank) and entry['type'] == 'PromptAnchorBank':
            module.load_state_dict(entry['state_dict'])
            loaded += 1
        elif isinstance(module, AnchorAwareFakeQuantize) and entry['type'] == 'AnchorAwareFakeQuantize':
            saved_scale = entry['scale'].to(module.scale.device)
            saved_zp = entry['zero_point'].to(module.zero_point.device)
            module.scale = torch.nn.Parameter(saved_scale)
            module.zero_point = saved_zp
            loaded += 1
    logger.info('Loaded PCSA checkpoint from {} ({}/{} modules matched)'.format(path, loaded, len(state)))


@torch.no_grad()
def calibrate(model, cali_data, BIG, pcsa_loaded=False):
    st = time.time()
    if BIG and not pcsa_loaded:
        model.extract_feat(cali_data[0])
        bimodal_adjust(model, logger=logger)
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')

    if pcsa_loaded:
        # Disable observers on AnchorAwareFakeQuantize so loaded scales are preserved
        for name, module in model.named_modules():
            if isinstance(module, AnchorAwareFakeQuantize):
                module.observer_enabled = 0
        logger.info('PCSA loaded from checkpoint -- freezing anchor-aware quantizers during activation calibration')

    for i in range(len(cali_data)):
        model.extract_feat(cali_data[i]) #HERE IS WHERE TO DO TIME EVALS
    # model.extract_feat(cali_data[0])
    rank, world_size = get_dist_info()
    observer = False
    if world_size!=1:
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                observer=True
                module.min_val.data /= world_size
                module.max_val.data /= world_size
                dist.all_reduce(module.min_val.data)
                dist.all_reduce(module.max_val.data)
        if not observer:
            for name, module in model.predictor.model.named_modules():
                if isinstance(module, ObserverBase):
                    observer=True
                    module.min_val.data /= world_size
                    module.max_val.data /= world_size
                    dist.all_reduce(module.min_val.data)
                    dist.all_reduce(module.max_val.data)

    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
    model.extract_feat(cali_data[0])

    ed = time.time()
    rank, _ = get_dist_info()
    if rank == 0:
        logger.info('the calibration time is {}'.format(ed - st))


def recon_model(model, fp_model, cali_data, recon_config, ahcptq_config, brecq=False,sam_version='sam2'):
    if len(cali_data) > 16:
        recon_config.keep_gpu = False
    
    if brecq:
        enable_quantization(model,'weight_fake_quant')
    else:
        enable_quantization(model)

    # start = time.time()
    # model.extract_feat(cali_data[0]) #HERE IS WHERE TO DO TIME EVALS
    # end = time.time()
    # print(f"Elapsed time: {end - start:.6f} seconds")
    
    def _recon_model(module, fp_module):
        if len(list(module.named_children()))==0 and hasattr(module, "model"): 
            module=module.model
            fp_module= fp_module.model
        for name, child_module in module.named_children():
            # print(name)
            
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)):
                # if isinstance(child_module, QuantDecoderOurAttentionBlock):
                logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                print("recon module", name)
                reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, recon_config, ahcptq_config)
            else:
                _recon_model(child_module, getattr(fp_module, name))
    # Start reconstruction
    if sam_version=='sam2': 
        _recon_model(model.predictor, fp_model.predictor)
    else: 
        _recon_model(model, fp_model)

if __name__ == '__main__':
    main()
