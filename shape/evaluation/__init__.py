"""
SHAPE Evaluation Module
评估模块
"""

from .hallucination_metrics import evaluate_pope, calculate_pope_metrics
from .benchmarks import run_benchmark_evaluation

__all__ = [
    "evaluate_pope",
    "calculate_pope_metrics",
    "run_benchmark_evaluation",
]
