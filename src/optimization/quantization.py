import torch
from bitsandbytes import functional as bf

class QuantizationManager:
    def __init__(self):
        self.quant_cache = {}
        
    def quantize_weights(self, model):
        # 4-bit block-wise quantization
        for name, param in model.named_parameters():
            quantized = bf.quantize_fp4(param.data)
            self.quant_cache[name] = quantized
            param.data = quantized.dequantize()
            
    def dynamic_requantize(self, layer):
        # Runtime requantization for memory-sensitive ops
        return bf.quantize_fp4(layer.weight)[0].dequantize()