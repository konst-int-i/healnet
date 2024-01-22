"""
Import relevant objects to benchmark from https://github.com/EPFLiGHT/MultiModN/
"""

from healnet.baselines.multimodn.encoders import MLPEncoder, ResNet, PatchEncoder
from healnet.baselines.multimodn.decoders import ClassDecoder, LogisticDecoder
from healnet.baselines.multimodn.multimodn import MultiModN

__all__ = [
    "MLPEncoder",
    "MLPFeatureEncoder",
    "ResNet",
    "ClassDecoder",
    "LogisticDecoder",
    "MultiModN"
]