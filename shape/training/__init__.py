"""
SHAPE Training Module
训练模块
"""

from .preference_trainer import PreferenceAlignmentTrainer
from .data_utils import create_preference_dataset, PreferenceDataCollator

__all__ = [
    "PreferenceAlignmentTrainer",
    "create_preference_dataset",
    "PreferenceDataCollator",
]
