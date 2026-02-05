"""
SHAPE Configuration Management System
配置管理系统
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    name_or_path: str = field(default="bczhou/tiny-llava-v1-hf")
    version: str = field(default="v1")
    vision_tower: str = field(default="openai/clip-vit-large-patch14-336")
    mm_projector_type: str = field(default="mlp2x_gelu")
    mm_vision_select_layer: int = field(default=-2)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)


@dataclass
class DataConfig:
    """数据集配置"""
    ocr_data_path: Optional[str] = field(default="./datasets/ocrvqa_answer_file_8k_dpo.jsonl")
    ocr_image_path: Optional[str] = field(default="data/ocrvqa/images/")
    textvqa_data_path: Optional[str] = field(default="./datasets/textvqa_answer_file_8k_dpo.jsonl")
    textvqa_image_path: Optional[str] = field(default="data/textvqa/train_images")
    image_aspect_ratio: str = field(default="pad")
    lazy_preprocess: bool = field(default=True)
    is_multimodal: bool = field(default=False)


@dataclass
class TrainingConfig:
    """训练配置"""
    # DPO 参数
    beta: float = field(default=0.1, metadata={"help": "DPO loss 的 beta 参数"})
    
    # 优化器配置
    learning_rate: float = field(default=2e-6)
    weight_decay: float = field(default=0.0)
    optimizer_type: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=0)
    warmup_ratio: float = field(default=0.03)
    max_grad_norm: float = field(default=1.0)
    
    # 训练参数
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    
    # 保存和日志
    output_dir: str = field(default="checkpoints/shape_model")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=8)
    logging_steps: int = field(default=1)
    evaluation_strategy: str = field(default="no")
    
    # DeepSpeed
    deepspeed_config: Optional[str] = field(default="configs/deepspeed/zero3_offload.json")
    
    # 精度设置
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    tf32: bool = field(default=True)
    
    # 其他
    model_max_length: int = field(default=2048)
    dataloader_num_workers: int = field(default=4)
    group_by_modality_length: bool = field(default=True)
    seed: int = field(default=42)
    
    # LoRA 配置
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    
    # WandB
    report_to: str = field(default="wandb")
    run_name: str = field(default="shape_preference_training")


@dataclass
class InferenceConfig:
    """推理配置"""
    big_model_path: str = field(default="PATH_TO_BASE_MODEL")
    small_model_path: str = field(default="PATH_TO_REWARD_MODEL")
    weight_big: float = field(default=0.7, metadata={"help": "基础模型权重"})
    weight_small: float = field(default=0.3, metadata={"help": "奖励模型权重"})
    device: str = field(default="cuda")
    batch_size: int = field(default=1)
    use_bf16: bool = field(default=True)


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path
