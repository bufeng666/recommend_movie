import torch
import numpy as np

class CIESS:
    """
    CIESS: Communication-Efficient Secure Scheme (Simplified Simulation)
    Focuses on compression (Sparsification + Quantization)
    """
    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio

    def compress(self, tensor):
        """
        Simulates compression by keeping only top-k gradients/weights
        and quantizing them.
        """
        # 1. Sparsification (Top-k)
        num_elements = tensor.numel()
        k = max(1, int(num_elements * self.compression_ratio))
        
        abs_tensor = torch.abs(tensor)
        values, indices = torch.topk(abs_tensor.view(-1), k)
        
        # 2. Quantization (Simulated 8-bit)
        # Scale to range [0, 255]
        min_val = values.min()
        max_val = values.max()
        
        if max_val == min_val:
            scale = 1.0
        else:
            scale = 255.0 / (max_val - min_val)
            
        quantized_values = torch.round((values - min_val) * scale).to(torch.uint8)
        
        # Return compressed package
        compressed_package = {
            'indices': indices,
            'quantized_values': quantized_values,
            'min_val': min_val,
            'scale': scale,
            'original_shape': tensor.shape
        }
        return compressed_package

    def decompress(self, compressed_package):
        """
        Reconstructs the tensor from the compressed package.
        """
        indices = compressed_package['indices']
        quantized_values = compressed_package['quantized_values']
        min_val = compressed_package['min_val']
        scale = compressed_package['scale']
        original_shape = compressed_package['original_shape']
        
        # De-quantize
        if scale == 0:
            values = quantized_values.float() # Should be zeros or min_val
        else:
            values = (quantized_values.float() / scale) + min_val
            
        # Reconstruct sparse tensor
        device = indices.device
        reconstructed = torch.zeros(original_shape, device=device).view(-1)
        reconstructed.scatter_(0, indices, values)
        
        return reconstructed.view(original_shape)
