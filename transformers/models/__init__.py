"""Model registry for transformer architectures"""

from .informer import InformerModel
from .encoder_only_transformer import EncoderOnlyTransformer
from .encoder_decoder_transformer import EncoderDecoderTransformer
from .lstm_transformer import LSTM32_TX
from .lstm_option_pricer import LSTMOptionPricer
# from .temporal_fusion_transformer import TemporalFusionTransformer

MODEL_REGISTRY = {
    'informer': InformerModel,
    'encoder_only_transformer': EncoderOnlyTransformer,
    'encoder_decoder_transformer': EncoderDecoderTransformer,
    'lstm32_tx': LSTM32_TX,
    'lstm_option_pricer': LSTMOptionPricer,
}

def get_model(model_name, **kwargs):
    """Get model class by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](**kwargs)