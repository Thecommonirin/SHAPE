"""
SHAPE Utility Functions
工具函数
"""
import os
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息
    
    Returns:
        包含设备信息的字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    
    return info


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
        precision: 小数精度
    
    Returns:
        格式化的指标字符串
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)


def ensure_dir_exists(path: str) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
    
    Returns:
        Path 对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数
    
    Args:
        model: PyTorch 模型
    
    Returns:
        参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
    }


def get_rank() -> int:
    """获取当前进程的 rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_main_process() -> bool:
    """判断是否为主进程"""
    return get_rank() == 0


def print_rank_0(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)
