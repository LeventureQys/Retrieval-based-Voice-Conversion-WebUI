"""
Inference module for RVC-LoRA

Contains inference scripts and model loaders.
"""

from .model_loader import (
    LoRAModelLoader,
    load_model_for_inference,
    get_model_info,
)

from .infer_lora import (
    LoRAInference,
    extract_features,
)

__all__ = [
    'LoRAModelLoader',
    'load_model_for_inference',
    'get_model_info',
    'LoRAInference',
    'extract_features',
]
