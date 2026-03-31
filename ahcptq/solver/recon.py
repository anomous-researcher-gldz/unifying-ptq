import numpy as np
import torch
import torch.nn as nn
import logging
from utils import DataSaverHook, StopForwardException
from ahcptq.quantization.quantized_module import QuantizedModule
from ahcptq.quantization.fake_quant import * #LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase , AdaptiveGranularityQuantize, GroupLSQFakeQuantize, HybridQuantize
logger = logging.getLogger('ahcptq')
import numpy as np
import torch

def get_img_np_from_cali_sample(cali_sample):
    """
    Extract an H x W x C numpy image from cali_sample['img'].

    Handles these common mmdet cases:
    - DataContainer with .data[0] a list of tensors
    - list/tuple of tensors
    - tensor of shape (B, C, H, W), (C, H, W), or (H, W, C)
    - numpy arrays of shape (H, W, C) or (H, W)
    """
    img = cali_sample["img"]

    # mmdet DataContainer: img.data[0] is usually a list of tensors
    if hasattr(img, "data"):
        img = img.data[0]

    # If it's a list/tuple, take the first element (batch size 1)
    if isinstance(img, (list, tuple)):
        if len(img) == 0:
            raise RuntimeError("Empty image list in cali_sample['img']")
        img = img[0]

    # Torch tensor cases
    if isinstance(img, torch.Tensor):
        t = img

        # If 4D, assume (B, C, H, W)
        if t.ndim == 4:
            t = t[0]  # (C, H, W) or (H, W, C)

        # Now expect 3D or 2D
        if t.ndim == 3:
            # If first dim looks like channels (1,3,4), treat as CHW
            if t.shape[0] in (1, 3, 4):
                # (C, H, W) -> (H, W, C)
                return t.permute(1, 2, 0).cpu().numpy()
            else:
                # Already (H, W, C)
                return t.cpu().numpy()
        elif t.ndim == 2:
            # Grayscale H x W -> H x W x 1
            return t.unsqueeze(-1).cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected tensor image shape: {t.shape}")

    # Numpy array cases
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            return img  # assume H x W x C
        elif img.ndim == 2:
            return img[..., None]  # H x W -> H x W x 1
        else:
            raise RuntimeError(f"Unexpected numpy image shape: {img.shape}")

    raise TypeError(f"Unsupported image type in cali_sample['img']: {type(img)}")
def run_predict_forward(model, cali_sample):
    """
    Drive SAM2 mask decoder so modules like mask_downsample are hit.

    `model` here is your detector wrapper (e.g. DetWrapperInstanceSAM2),
    which has `predictor` inside.
    """
    # predictor = model.predictor  # adjust if your wrapper names differ
    # print(hasattr(model, "predictor"))
    predictor = model.predictor if hasattr(model, "predictor") else model
    # predictor=model 
    # ---- get image from cali_sample ----
    # This depends on how your cali_data is structured.
    # Common mmdet pattern: cali_sample['img'].data[0] is a tensor (1, C, H, W)
    img = cali_sample["img"]
    if hasattr(img, "data"):
        img_tensor = img.data[0]  # (1, C, H, W)
    else:
        img_tensor = img  # already a tensor?
    
    img_np = get_img_np_from_cali_sample(cali_sample) # (H, W, C)
    if img_np.dtype != np.uint8:
        img_vis = img_np.astype(np.float32)

        # If it looks normalized [0,1], scale up
        mx = img_vis.max()
        mn = img_vis.min()
        if mx <= 1.0 and mn >= 0.0:
            img_vis = img_vis * 255.0

        img_np = np.clip(img_vis, 0, 255).astype(np.uint8)
    # ---- set image ----
    predictor.set_image(img_np)

    # ---- dummy point prompt in center ----
    h, w, _ = img_np.shape
    point_coords = np.array([[w / 2.0, h / 2.0]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # foreground

    # This should run the mask decoder (mask_downsample etc.)
    _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
def save_inp_oup_data(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):

    device = next(model.parameters()).device
    # print(device)
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            # print(keep_gpu,cali_data[i].keys() )
            try:
                _ = model.extract_feat(cali_data[i])
            except StopForwardException:
                pass

            if store_inp and not data_saver.input_store:
                try:
                    run_predict_forward(model, cali_data[i])
                except StopForwardException:
                    pass

            # -------- 3) still nothing? skip this sample --------
            if store_inp and not data_saver.input_store:
                # module not hit by either path for this sample
                continue
            if store_oup and data_saver.output_store is None:
                continue
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0])
                else:
                    input_data = data_saver.input_store[0]
                    if isinstance(input_data,tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                        else:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                out = data_saver.output_store
                # print(out)
                # SAM2 sometimes returns dict / tuple / list from modules.
                if isinstance(out, dict):
                    # pick the first tensor-like value
                    out = out['vision_features']
                if isinstance(out, (list, tuple)):
                    # take the first element (e.g., (queries, keys), (masks, iou, ...)
                    out = out[0]
                
                if keep_gpu:
                    cached[1].append(out.detach())
                else:
                    cached[1].append(out.detach().cpu())
    # if store_inp:
    #     cached[0] = torch.cat([x for x in cached[0]])
    # if store_oup:
    #     cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 reg_weight=None,
                 reg_weight_lamb=0.1
                 ):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.reg_weight=reg_weight
        self.reg_weight_lamb = reg_weight_lamb

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
            w_reg_loss = 0
        else:
            round_loss = 0
            w_reg_loss = 0
            layer_len = 0
            for name, layer in self.module.named_modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)) and "patch_embed" not in name:
                    if self.reg_weight is None:
                        round_vals = layer.weight_fake_quant.rectified_sigmoid()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        # total_loss = w_reg_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.4f} (rec:{:.4f}, round:{:.4f}, rw:{:.4f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), float(w_reg_loss), b, self.count))
        return total_loss


def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    # print(pred)
    if isinstance(pred,dict):
        pred = pred['vision_features'] 

    return (pred - tgt).abs().pow(p).sum(1).mean()


def reconstruction(model, fp_model, module, fp_module, cali_data, config, ahcptq_config):
    device = next(module.parameters()).device
    # get data first
    # print(config)
    #where to do time evaluation
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    if len(quant_inp) == 0 or len(fp_inp) == 0 or len(fp_oup) == 0:
        logger.info(f"Module {module} never hit even with predict fallback; skipping reconstruction.")
        return
    w_para, a_para = [], []
    
    # # for the bimodal block, add the gamma parameter
    # gamma_para = []
    # if hasattr(module,'gamma') and config.gamma_tune:
    #     gamma_para.append(module.gamma)
    # print("here", module)
    for name, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (config.only4 and ('k_proj' in name or 'q_proj' in name))
        print(('only4' not in config.keys()))
        if isinstance(layer, (nn.Linear, nn.Conv2d)) and 'patch_embed' not in name:
            # print(name)
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
            # w_para += [weight_quantizer.scale]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, AnchorAwareFakeQuantize): 
                    a_para += [layer.scale]
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                    # a_para += [layer.scale_tail]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, GroupLSQFakeQuantize):
                    a_para += [layer.grouped_scales]
                    # a_para += [layer.grouped_scales_tail]
                if isinstance(layer, HybridQuantize):
                    a_para += [layer.scale_log]
                    a_para += [layer.scale_uni]
    
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        # a_opt = torch.optim.Adam([{"params":a_para,"lr":config.scale_lr},{"params":module.gamma,"lr":config.scale_lr}])
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    # if len(gamma_para) != 0:
    #     gamma_opt = torch.optim.Adam(gamma_para, lr=config.gamma_lr)
    # else:
    #     gamma_opt = None
    
    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))


    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp,fp_oup,quant_inp
        torch.cuda.empty_cache()
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up)

    from mmdet.utils import build_ddp,build_dp
    import os
    module_ddp = build_dp(module, 'cuda', device_ids=[0])
    # module_ddp = build_ddp(
    #     module,
    #     'cuda',
    #     device_ids=[int(os.environ['LOCAL_RANK'])],
    #     broadcast_buffers=False)  
    try:
        for i in range(len(fp_oup)):
            fp_oup[i] = fp_oup[i].cuda()

        for i,t in enumerate(fp_inp):
            if isinstance(t,tuple):
                if len(t) == 3:
                    fp_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                else:
                    fp_inp[i] = (t[0].cuda(),t[1].cuda())
            else:
                fp_inp[i] = t.cuda()

        for i,t in enumerate(quant_inp):
            if isinstance(t,tuple):
                if len(t) == 3:
                    quant_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                else:
                    quant_inp[i] = (t[0].cuda(),t[1].cuda())
            else:
                quant_inp[i] = t.cuda()
    except:
        in_cpu = 32
        logger.info('in_cpu 32')
        for i in range(len(fp_oup)):
            fp_oup[i] = fp_oup[i].cpu()

        for i,t in enumerate(fp_inp):
            if i < in_cpu:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        fp_inp[i] = (t[0].cpu(),t[1].cpu(),t[2].cpu())
                    else:
                        fp_inp[i] = (t[0].cpu(),t[1].cpu())
                else:
                    fp_inp[i] = t.cpu()
            else:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        fp_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                    else:
                        fp_inp[i] = (t[0].cuda(),t[1].cuda())
                else:
                    fp_inp[i] = t.cuda()

        for i,t in enumerate(quant_inp):
            if i < in_cpu:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        quant_inp[i] = (t[0].cpu(),t[1].cpu(),t[2].cpu())
                    else:
                        quant_inp[i] = (t[0].cpu(),t[1].cpu())
                else:
                    quant_inp[i] = t.cpu()
            else:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        quant_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                    else:
                        quant_inp[i] = (t[0].cuda(),t[1].cuda())
                else:
                    quant_inp[i] = t.cuda()


    sz = len(cali_data)
    for i in range(config.iters):
        idx = torch.randint(0, sz, (1, ))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]
    
            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda()
                cur_fp_inp = cur_fp_inp.cuda()
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp.cuda())
            elif len(cur_quant_inp) == 2:
                
                cur_quant_inp = (cur_quant_inp[0].cuda(), cur_quant_inp[1].cuda())
                cur_fp_inp = (cur_fp_inp[0].cuda(), cur_fp_inp[1].cuda())
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0,cur_inp1)
            else:
                cur_quant_inp = (cur_quant_inp[0].cuda(), cur_quant_inp[1].cuda(), cur_quant_inp[2].cuda())
                cur_fp_inp = (cur_fp_inp[0].cuda(), cur_fp_inp[1].cuda(), cur_fp_inp[2].cuda())
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2], cur_fp_inp[2])
                cur_inp = (cur_inp0,cur_inp1,cur_inp2)
        else:
            cur_inp = quant_inp[idx]
        cur_fp_oup = fp_oup[idx].cuda()
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        cur_quant_oup = module_ddp(cur_inp)
        err = loss_func(cur_quant_oup, cur_fp_oup)
        # print(err)
        cur_inp = None
        cur_quant_oup = None
        torch.cuda.empty_cache()
        err.backward() # del cur_inp cur_quant_oup
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

        if ahcptq_config.cag:
            contains_group_lsq = any(isinstance(sub_module, GroupLSQFakeQuantize) for sub_module in module.modules())
            if contains_group_lsq:
                if i in [int(config.iters * 0.2), int(config.iters * 0.4), int(config.iters * 0.6), int(config.iters * 0.8)]:
                    if i == int(config.iters * 0.2):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcptq_config.group*8)
                    if i == int(config.iters * 0.4):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcptq_config.group*4)
                    if i == int(config.iters * 0.6):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcptq_config.group*2)
                    if i == int(config.iters * 0.8):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcptq_config.group)
    
    del fp_inp,fp_oup,quant_inp,cur_fp_oup
    torch.cuda.empty_cache()
    
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0

def group_channel(module, a_para, config, num_channel):
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, GroupLSQFakeQuantize):
            sub_module.group_channel(num_channel)
            logger.info(f'Group number of activation channel into {num_channel}')
            a_para += [sub_module.grouped_scales]
            if len(a_para) != 0:
                a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
                a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
            else:
                a_opt, a_scheduler = None, None
    return a_opt, a_scheduler
