from ...import_utils import is_bnb_4bit_available, is_bnb_available

from .config import PruneLoraConfig
# from .gptq import QuantLinear
# from .layer import Conv2d, Embedding, Linear, LoraLayer
from .model import PruneLoraModel


__all__ = ["PruneLoraConfig", "PruneLoraModel"]


# def __getattr__(name):
#     if (name == "Linear8bitLt") and is_bnb_available():
#         from .bnb import Linear8bitLt

#         return Linear8bitLt

#     if (name == "Linear4bit") and is_bnb_4bit_available():
#         from .bnb import Linear4bit

#         return Linear4bit

#     raise AttributeError(f"module {__name__} has no attribute {name}")
