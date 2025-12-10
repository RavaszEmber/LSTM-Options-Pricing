"""Model registry for transformer architectures"""

from .pimentel_mlp import PimentelMLP
# from .temporal_fusion_transformer import TemporalFusionTransformer

MODEL_REGISTRY = {
    'pimentel_mlp': PimentelMLP,
}

def get_model(model_name, **kwargs):
    """Get model class by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](**kwargs)