"""
SHAPE Hallucination Evaluation Metrics
幻觉评估指标 - POPE (Polling-based Object Probing Evaluation)
"""
import json
from typing import Dict, List, Any
from pathlib import Path


def evaluate_pope(predictions: List[Dict[str, Any]], annotations: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    评估 POPE 指标
    
    Args:
        predictions: 预测结果列表
        annotations: 标注结果列表
    
    Returns:
        评估指标字典
    """
    assert len(predictions) == len(annotations), "预测和标注数量不匹配"
    
    tp = 0  # True Positive
    tn = 0  # True Negative
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    for pred, anno in zip(predictions, annotations):
        pred_label = pred.get("label", "").lower()
        true_label = anno.get("label", "").lower()
        
        # 将答案转换为二分类
        pred_binary = 1 if "yes" in pred_label else 0
        true_binary = 1 if "yes" in true_label else 0
        
        if pred_binary == 1 and true_binary == 1:
            tp += 1
        elif pred_binary == 0 and true_binary == 0:
            tn += 1
        elif pred_binary == 1 and true_binary == 0:
            fp += 1
        else:
            fn += 1
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Yes 率（衡量模型是否过度肯定）
    yes_rate = (tp + fp) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": yes_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    
    return metrics


def calculate_pope_metrics(result_file: str, annotation_file: str) -> Dict[str, float]:
    """
    从文件计算 POPE 指标
    
    Args:
        result_file: 结果文件路径
        annotation_file: 标注文件路径
    
    Returns:
        评估指标字典
    """
    # 读取预测结果
    with open(result_file, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f]
    
    # 读取标注
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = [json.loads(line) for line in f]
    
    return evaluate_pope(predictions, annotations)


def print_pope_metrics(metrics: Dict[str, float]):
    """打印 POPE 指标"""
    print("=" * 80)
    print("POPE 评估结果:")
    print("=" * 80)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1 分数:            {metrics['f1']:.4f}")
    print(f"Yes 率:             {metrics['yes_rate']:.4f}")
    print("-" * 80)
    print(f"TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print("=" * 80)
