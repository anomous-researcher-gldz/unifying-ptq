import logging
from .fake_quant import QuantizeBase
from .fake_quant_blocks import * 
logger = logging.getLogger("ahcptq")

def is_sam2_predictor(predictor) -> bool:
    if predictor is None:
        return False
    t = type(predictor)
    name = t.__name__.lower()
    mod  = (t.__module__ or "").lower()
    # print(t,name,mod)
    return ("sam2" in name) or ("sam2" in mod)

def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    if is_sam2_predictor(model): 
        # print("here")
        model = model.predictor.model 
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase) or isinstance(submodule,BlockQuantizeBase):
            # if 'weight' in quantizer_type: 
            # print("here")
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type='fake_quant'):
    if is_sam2_predictor(model): 
        model = model.predictor.model 
    logger.info('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase) or isinstance(submodule,BlockQuantizeBase):
            # print("here")
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    logger.info('Disable observer and disable quantize.')
    if is_sam2_predictor(model): 
        model = model.predictor.model 
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase) or isinstance(submodule,BlockQuantizeBase):
            logger.debug('Disable observer and disable quant: {}'.format(name))
            print(name)
            submodule.disable_observer()
            submodule.disable_fake_quant()
