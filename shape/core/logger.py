"""
SHAPE Logging System
日志系统
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "shape",
    log_level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志名称
        log_level: 日志级别
        log_dir: 日志目录（如果为 None 则不保存到文件）
        console: 是否输出到控制台
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # 清除已有的处理器
    
    # 格式化器
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件保存至: {log_file}")
    
    return logger


def get_logger(name: str = "shape") -> logging.Logger:
    """获取已配置的日志记录器"""
    return logging.getLogger(name)


def log_training_config(logger: logging.Logger, config: dict):
    """记录训练配置"""
    logger.info("=" * 80)
    logger.info("训练配置:")
    logger.info("=" * 80)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
