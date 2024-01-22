"""
Import relevant objects to benchmark from https://github.com/EPFLiGHT/MultiModN/
"""

from healnet.baselines.multimodn.encoders import MLPEncoder, ResNet, RNNEncoder
# from healnet.baselines.multimodn.resnet_encoder import ResNet
from healnet.baselines.multimodn.decoders import ClassDecoder, LogisticDecoder

__all__ = [
    "MLPEncoder",
    "MLPFeatureEncoder",
    "ResNet",
    "ClassDecoder",
    "LogisticDecoder"
]