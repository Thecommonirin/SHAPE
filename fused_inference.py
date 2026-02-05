#!/usr/bin/env python3
"""
SHAPE Fused Inference Script
融合推理脚本

使用 Token-wise 融合方法，结合基础模型和奖励模型进行推理
"""
import gc
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

from shape.core import setup_logger, get_logger, set_seed, format_metrics


class FusedInferenceEngine:
    """融合推理引擎"""
    
    def __init__(
        self,
        base_model_path: str,
        reward_model_path: str,
        weight_base: float = 0.7,
        weight_reward: float = 0.3,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        """
        初始化融合推理引擎
        
        Args:
            base_model_path: 基础模型路径（如 LLaVA-1.5-7b）
            reward_model_path: 奖励模型路径（如 tiny-llava）
            weight_base: 基础模型权重
            weight_reward: 奖励模型权重
            device: 设备
            use_bf16: 是否使用 bf16
        """
        self.logger = get_logger("shape.inference")
        
        self.base_model_path = base_model_path
        self.reward_model_path = reward_model_path
        self.weight_base = weight_base
        self.weight_reward = weight_reward
        self.device = device
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        self.logger.info(f"初始化融合推理引擎...")
        self.logger.info(f"  基础模型: {base_model_path}")
        self.logger.info(f"  奖励模型: {reward_model_path}")
        self.logger.info(f"  基础模型权重: {weight_base}")
        self.logger.info(f"  奖励模型权重: {weight_reward}")
        self.logger.info(f"  设备: {device}")
        self.logger.info(f"  精度: {'BF16' if use_bf16 else 'FP16'}")
        
        # 加载模型和处理器
        self._load_models()
    
    def _load_models(self):
        """加载模型"""
        self.logger.info("加载基础模型...")
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype
        )
        self.base_processor = AutoProcessor.from_pretrained(self.base_model_path)
        
        self.logger.info("加载奖励模型...")
        self.reward_model = AutoModelForVision2Seq.from_pretrained(
            self.reward_model_path,
            torch_dtype=self.dtype
        )
        self.reward_processor = AutoProcessor.from_pretrained(self.reward_model_path)
        
        self.logger.info("模型加载完成!")
    
    def infer(self, image: Image.Image, question: str) -> str:
        """
        执行融合推理
        
        Args:
            image: 输入图像
            question: 输入问题
        
        Returns:
            预测答案
        """
        try:
            # 基础模型推理
            self.base_model.to(self.device)
            inputs_base = self.base_processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device, self.dtype)
            
            with torch.no_grad(), autocast(device_type='cuda', dtype=self.dtype):
                out_base = self.base_model(**inputs_base, return_dict=True)
            
            logits_base = out_base.logits[:, -1, :].cpu()
            self.base_model.to("cpu")
            del inputs_base, out_base
            torch.cuda.empty_cache()

            # 奖励模型推理
            self.reward_model.to(self.device)
            inputs_reward = self.reward_processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device, self.dtype)
            
            with torch.no_grad(), autocast(device_type='cuda', dtype=self.dtype):
                out_reward = self.reward_model(**inputs_reward, return_dict=True)
            
            logits_reward = out_reward.logits[:, -1, :].cpu()
            self.reward_model.to("cpu")
            del inputs_reward, out_reward
            torch.cuda.empty_cache()

            # 融合 logits
            fused_logits = self.weight_base * logits_base + self.weight_reward * logits_reward
            top_token_id = torch.argmax(fused_logits, dim=-1)
            answer = self.base_processor.tokenizer.decode(top_token_id[0], skip_special_tokens=True)
            
            return answer.strip()

        except Exception as e:
            self.logger.error(f"推理失败: {str(e)}")
            return f"[ERROR] {str(e)}"
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    def batch_infer(self, image_paths: list, questions: list) -> list:
        """
        批量推理
        
        Args:
            image_paths: 图像路径列表
            questions: 问题列表
        
        Returns:
            答案列表
        """
        results = []
        total = len(image_paths)
        
        for i, (img_path, question) in enumerate(zip(image_paths, questions)):
            self.logger.info(f"处理 {i+1}/{total}: {img_path}")
            
            try:
                image = Image.open(img_path).convert("RGB")
                answer = self.infer(image, question)
                results.append({
                    "image_path": img_path,
                    "question": question,
                    "answer": answer,
                })
            except Exception as e:
                self.logger.error(f"处理失败: {str(e)}")
                results.append({
                    "image_path": img_path,
                    "question": question,
                    "answer": f"[ERROR] {str(e)}",
                })
        
        return results


def evaluate_on_dataset(
    engine: FusedInferenceEngine,
    dataset_name: str = "MM-Vet/mm-vet",
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    在数据集上评估
    
    Args:
        engine: 融合推理引擎
        dataset_name: 数据集名称
        split: 数据集分割
        max_samples: 最大样本数
    
    Returns:
        评估结果
    """
    logger = get_logger("shape.inference")
    logger.info(f"加载数据集: {dataset_name} ({split})")
    
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"数据集样本数: {len(dataset)}")
    
    correct = 0
    total = 0
    results = []
    
    for idx, item in enumerate(dataset):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            logger.warning(f"无法加载图像: {item.get('image_path', 'N/A')}")
            continue
        
        question = item["question"]
        gold_answer = item["answer"]
        
        pred = engine.infer(image, question)
        
        logger.info(f"[{idx+1}/{len(dataset)}]")
        logger.info(f"  问题: {question}")
        logger.info(f"  预测: {pred}")
        logger.info(f"  答案: {gold_answer}")
        
        # 简单的准确率计算（实际应用中可能需要更复杂的匹配）
        is_correct = pred.lower() in [a.lower() for a in gold_answer] if isinstance(gold_answer, list) else pred.lower() == gold_answer.lower()
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "prediction": pred,
            "gold_answer": gold_answer,
            "correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("评估结果:")
    logger.info(f"  准确率: {accuracy * 100:.2f}%")
    logger.info(f"  正确数: {correct}/{total}")
    logger.info("=" * 80)
    
    return {
        "metrics": metrics,
        "results": results,
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SHAPE 融合推理")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="基础模型路径（如 LLaVA-1.5-7b）"
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        required=True,
        help="奖励模型路径（如 tiny-llava）"
    )
    parser.add_argument(
        "--weight-base",
        type=float,
        default=0.7,
        help="基础模型权重"
    )
    parser.add_argument(
        "--weight-reward",
        type=float,
        default=0.3,
        help="奖励模型权重"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="使用 BF16 精度"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MM-Vet/mm-vet",
        help="评估数据集"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="数据集分割"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果路径"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    setup_logger("shape", console=True)
    logger = get_logger("shape.inference")
    
    # 创建推理引擎
    engine = FusedInferenceEngine(
        base_model_path=args.base_model,
        reward_model_path=args.reward_model,
        weight_base=args.weight_base,
        weight_reward=args.weight_reward,
        device=args.device,
        use_bf16=args.use_bf16,
    )
    
    # 在数据集上评估
    results = evaluate_on_dataset(
        engine=engine,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )
    
    # 保存结果
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
