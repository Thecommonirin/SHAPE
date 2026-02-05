"""
SHAPE Preference Alignment Trainer
偏好对齐训练器 - 基于 DPO 的实现
"""

from .llava_dpo_trainer import LlavaDPOTrainer

# 为了更好的语义，重命名为 PreferenceAlignmentTrainer
PreferenceAlignmentTrainer = LlavaDPOTrainer

__all__ = ["PreferenceAlignmentTrainer"]
