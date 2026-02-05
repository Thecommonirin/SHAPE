#!/usr/bin/env python3
"""
SHAPE Preference Alignment Training Script
偏好对齐训练脚本

使用 DPO (Direct Preference Optimization) 方法训练视觉-语言模型
"""
import os
os.environ["WANDB_PROJECT"] = "shape"

import sys
import json
import copy
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image

import transformers
from transformers import TrainerCallback, HfArgumentParser, TrainingArguments

from src.llava.model import *
from src.llava.constants import IGNORE_INDEX
from src.llava import conversation as conversation_lib
from src.llava.train.train import preprocess_multimodal, preprocess

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from peft.peft_model import PeftModelForCausalLM

# 导入 SHAPE 模块
from shape.core import setup_logger, get_logger, set_seed, get_device_info, format_metrics, print_rank_0, count_parameters
from shape.training import PreferenceAlignmentTrainer
from configs.config import ModelConfig, DataConfig, TrainingConfig


logger = get_logger("shape.training")


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: Optional[str] = field(default="bczhou/tiny-llava-v1-hf")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    """数据参数"""
    ocr_data_path: str = field(default=None, metadata={"help": "OCR-VQA 数据路径"})
    ocr_image_path: str = field(default=None, metadata={"help": "OCR-VQA 图像路径"})
    textvqa_data_path: str = field(default=None, metadata={"help": "TextVQA 数据路径"})
    textvqa_image_path: str = field(default=None, metadata={"help": "TextVQA 图像路径"})
    lazy_preprocess: bool = field(default=False)
    is_multimodal: bool = field(default=False)
    image_folder: Optional[str] = field(default="")
    image_aspect_ratio: str = field(default='pad')


@dataclass
class ScriptArguments:
    """脚本参数"""
    # 基础参数
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=2048)
    
    # 量化参数
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    
    # LoRA 参数
    lora_enable: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_weight_path: Optional[str] = field(default=None)
    lora_bias: Optional[str] = field(default="none")
    mm_projector_lr: Optional[float] = field(default=None)
    group_by_modality_length: Optional[bool] = field(default=False)
    
    # DPO 参数
    beta: Optional[float] = field(default=0.1, metadata={"help": "DPO loss 的 beta 参数"})
    
    # 训练参数
    learning_rate: Optional[float] = field(default=2e-6)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    warmup_steps: Optional[int] = field(default=0)
    weight_decay: Optional[float] = field(default=0.0)
    optimizer_type: Optional[str] = field(default="adamw_torch")
    max_grad_norm: Optional[float] = field(default=1.0)
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    ddp_find_unused_parameters: Optional[bool] = field(default=True)
    max_prompt_length: Optional[int] = field(default=512)
    max_length: Optional[int] = field(default=1024)
    max_steps: Optional[int] = field(default=-1)
    logging_steps: Optional[int] = field(default=1)
    save_steps: Optional[int] = field(default=500)
    evaluation_strategy: Optional[str] = field(default='no')
    eval_steps: Optional[int] = field(default=-1)
    output_dir: Optional[str] = field(default="./checkpoints")
    deepspeed: Optional[str] = field(default=None)
    bf16: Optional[bool] = field(default=True)
    fp16: Optional[bool] = field(default=False)
    num_train_epochs: Optional[int] = field(default=1)
    save_strategy: Optional[str] = field(default="steps")
    save_total_limit: Optional[int] = field(default=8)
    warmup_ratio: Optional[float] = field(default=0.03)
    tf32: Optional[bool] = field(default=True)
    dataloader_num_workers: Optional[int] = field(default=4)
    fsdp: Optional[str] = field(default='')
    local_rank: int = field(default=-1)
    seed: Optional[int] = field(default=42)
    
    # 日志参数
    report_to: Optional[str] = field(default="wandb")
    run_name: Optional[str] = field(default="shape_preference_training")
    
    # Debug 参数
    ignore_bias_buffers: Optional[bool] = field(default=False)


def rank0_print(*args):
    """仅在主进程打印"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def find_all_linear_names(model):
    """查找所有线性层名称（用于 LoRA）"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class PreferenceDataset(Dataset):
    """偏好对数据集"""

    def __init__(
        self,
        ocr_data_path: str,
        ocr_image_path: str,
        textvqa_data_path: str,
        textvqa_image_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(PreferenceDataset, self).__init__()
        
        list_data_dict = []
        
        # 加载 OCR-VQA 数据
        if ocr_data_path is not None:
            with open(ocr_data_path) as f:
                ocr_data = f.readlines()
            ocr_data_dict = self.preprocess_data(ocr_data, ocr_image_path)
            list_data_dict += ocr_data_dict * 2  # 重复采样以平衡数据
            rank0_print(f"加载 OCR-VQA 数据: {len(ocr_data_dict)} 样本")

        # 加载 TextVQA 数据
        if textvqa_data_path is not None:
            with open(textvqa_data_path) as f:
                text_data = f.readlines()
            text_data_dict = self.preprocess_data(text_data, textvqa_image_path)
            list_data_dict += text_data_dict * 2
            rank0_print(f"加载 TextVQA 数据: {len(text_data_dict)} 样本")

        rank0_print(f"总计 {len(list_data_dict)} 个训练样本")
        
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def preprocess_data(self, data_list, image_path):
        """预处理数据"""
        processed_data = []
        for line in data_list:
            data = json.loads(line)
            image_id = data["image_id"]
            chosen = data["chosen"]
            reject = data["reject"]
            question = "<image>\n" + data["question"]
            
            processed_data.append({
                "id": image_id,
                "image": os.path.join(image_path, image_id),
                "chosen_conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": chosen},
                ],
                "reject_conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": reject},
                ],
            })
        return processed_data

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['chosen_conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        
        # 处理图像
        if 'image' in self.list_data_dict[i]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            image = Image.open(image_file).convert('RGB')
            
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                
                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            chosen_sources = preprocess_multimodal(
                copy.deepcopy([e["chosen_conversations"] for e in sources]),
                self.data_args
            )
            reject_sources = preprocess_multimodal(
                copy.deepcopy([e["reject_conversations"] for e in sources]),
                self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # 预处理文本
        chosen_data_dict = preprocess(
            chosen_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )
        reject_data_dict = preprocess(
            reject_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )
        
        if isinstance(i, int):
            data_dict = dict(
                chosen_input_ids=chosen_data_dict["input_ids"][0],
                chosen_labels=chosen_data_dict["labels"][0],
                reject_input_ids=reject_data_dict["input_ids"][0],
                reject_labels=reject_data_dict["labels"][0],
            )

        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        return data_dict


@dataclass
class DataCollatorForPreferenceDataset(object):
    """偏好数据集的数据整理器"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels = tuple(
            [instance[key] for instance in instances]
            for key in ("chosen_input_ids", "chosen_labels", "reject_input_ids", "reject_labels")
        )
        
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            chosen_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        reject_input_ids = torch.nn.utils.rnn.pad_sequence(
            reject_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        reject_labels = torch.nn.utils.rnn.pad_sequence(
            reject_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        
        chosen_input_ids = chosen_input_ids[:, :self.tokenizer.model_max_length]
        chosen_labels = chosen_labels[:, :self.tokenizer.model_max_length]
        reject_input_ids = reject_input_ids[:, :self.tokenizer.model_max_length]
        reject_labels = reject_labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
) -> Dict:
    """创建监督学习数据模块"""
    train_dataset = PreferenceDataset(
        ocr_data_path=data_args.ocr_data_path,
        ocr_image_path=data_args.ocr_image_path,
        textvqa_data_path=data_args.textvqa_data_path,
        textvqa_image_path=data_args.textvqa_image_path,
        tokenizer=tokenizer,
        data_args=data_args
    )
    data_collator = DataCollatorForPreferenceDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    """处理 DeepSpeed ZeRO-3 参数"""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    """获取 PEFT 模型状态（ZeRO-3 兼容）"""
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """获取非 LoRA 参数状态（ZeRO-3 兼容）"""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """获取多模态适配器状态（ZeRO-3 兼容）"""
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """安全保存模型"""
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


class SaverCallback(TrainerCallback):
    """训练结束时保存模型的回调"""
    
    def on_train_end(self, args, state, control, **kwargs):
        if isinstance(kwargs['model'], PeftModelForCausalLM):
            torch.cuda.synchronize()
            state_dict = get_peft_state_maybe_zero_3(
                kwargs['model'].named_parameters(), "none"
            )
            kwargs['model'].save_pretrained(args.output_dir)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                kwargs['model'].named_parameters()
            )
            kwargs['model'].config.save_pretrained(args.output_dir)
            kwargs['model'].save_pretrained(args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(args.output_dir, 'non_lora_trainables.bin'))


def setup_llava_model(model_args, data_args, script_args):
    """设置 LLaVA 模型"""
    # 确定设备
    if "LOCAL_RANK" not in os.environ:
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    
    compute_dtype = (
        torch.float16 if script_args.fp16 else 
        (torch.bfloat16 if script_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if script_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": device},
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=script_args.bits == 4,
                load_in_8bit=script_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=script_args.double_quant,
                bnb_4bit_quant_type=script_args.quant_type
            )
        ))

    # 加载模型
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = script_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if script_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if script_args.fp16 else 
            (torch.bfloat16 if script_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

    if script_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if script_args.lora_enable:
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if script_args.bits == 16:
            if script_args.bf16:
                model.to(torch.bfloat16)
            if script_args.fp16:
                model.to(torch.float16)
        rank0_print("添加 LoRA 适配器...")
        model = get_peft_model(model, lora_config)

    # 设置tokenizer
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 初始化视觉模块
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=script_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
            device=device
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = script_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = script_args.freeze_mm_mlp_adapter
        if script_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if script_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = script_args.mm_projector_lr
        script_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if script_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if script_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if script_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
                        
    return model, tokenizer


def main():
    """主函数"""
    # 解析参数
    parser = transformers.HfArgumentParser(
        (ScriptArguments, ModelArguments, DataArguments)
    )
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(script_args.seed)
    
    # 设置日志
    setup_logger(
        name="shape",
        log_level=logging.INFO,
        log_dir=os.path.join(script_args.output_dir, "logs"),
        console=True,
    )
    
    # 打印设备信息
    device_info = get_device_info()
    rank0_print("=" * 80)
    rank0_print("设备信息:")
    for key, value in device_info.items():
        rank0_print(f"  {key}: {value}")
    rank0_print("=" * 80)
    
    # 设置 policy 模型
    rank0_print("\n初始化 policy 模型...")
    llava_policy_model, tokenizer = setup_llava_model(
        model_args=model_args,
        data_args=data_args,
        script_args=script_args,
    )
    
    # 打印模型参数信息
    param_stats = count_parameters(llava_policy_model)
    rank0_print(f"模型参数统计:")
    rank0_print(f"  总参数: {param_stats['total']:,}")
    rank0_print(f"  可训练参数: {param_stats['trainable']:,}")
    rank0_print(f"  冻结参数: {param_stats['frozen']:,}")
    rank0_print(f"  可训练比例: {param_stats['trainable_ratio']:.2%}")
    
    # 设置 reference 模型
    rank0_print("\n初始化 reference 模型...")
    script_args.lora_enable = False
    llava_ref_model, _ = setup_llava_model(
        model_args=model_args,
        data_args=data_args,
        script_args=script_args,
    )
    
    # 冻结 reference 模型
    for n, p in llava_ref_model.named_parameters():
        p.requires_grad = False
    
    # 准备数据
    rank0_print("\n准备数据...")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args
    )
    
    # 配置训练参数
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False
    
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=script_args.ddp_find_unused_parameters,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        deepspeed=script_args.deepspeed,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        save_total_limit=script_args.save_total_limit,
        warmup_ratio=script_args.warmup_ratio,
        tf32=script_args.tf32,
        dataloader_num_workers=script_args.dataloader_num_workers,
        fp16=script_args.fp16,
        seed=script_args.seed,
    )

    # 初始化偏好对齐训练器
    rank0_print("\n初始化偏好对齐训练器...")
    rank0_print(f"DPO Beta 参数: {script_args.beta}")
    
    dpo_trainer = PreferenceAlignmentTrainer(
        model=llava_policy_model,
        ref_model=llava_ref_model,
        args=training_args,
        beta=script_args.beta,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        **data_module,
    )

    dpo_trainer.add_callback(SaverCallback())
    
    # 开始训练
    rank0_print("\n" + "=" * 80)
    rank0_print("开始训练...")
    rank0_print("=" * 80 + "\n")
    
    dpo_trainer.train()
    
    # 保存模型
    rank0_print("\n保存模型...")
    safe_save_model_for_hf_trainer(trainer=dpo_trainer, output_dir=script_args.output_dir)
    
    rank0_print("\n" + "=" * 80)
    rank0_print("训练完成!")
    rank0_print("=" * 80)


if __name__ == "__main__":
    main()
