# SHAPE: Self-Improved Holistic Alignment for Preference Enhancement

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

## 📖 Introduction

**SHAPE** is an innovative preference alignment framework for Large Vision-Language Models (LVLMs). It focuses on mitigating hallucinations and enhancing model reliability through **Direct Preference Optimization (DPO)** without relying on expensive human annotations.

### ✨ Key Features

- **Self-Supervised Alignment**: Transforms existing supervised image-text pairs into preference tuplets via visual augmentation and summarization.
- **DPO Training**: Efficient preference learning without complex Reinforcement Learning (RL) pipelines.
- **Reward Model Guidance**: Utilizes a lightweight model (e.g., Tiny-LLaVA) as a reward signal provider.
- **Fused Inference**: Token-wise fusion of the base model and reward model outputs for robust generation.
- **Hallucination Mitigation**: Significantly reduces hallucinations on benchmarks like POPE, OCR-VQA, and TextVQA.
- **Modular Design**: Clean, extensible code structure for easy customization.

### 🔍 Comparison with Other Methods

| Feature | SHAPE (Ours) | TITA | SeVa |
| :--- | :---: | :---: | :---: |
| **Data Generation** | **Reward-Guided / Summarization** | Iterative Training | Image Augmentation |
| **Training Method** | **DPO** | Hybrid PPO/DPO | DPO |
| **Inference** | **Fused / Holistic** | Single Model | Single Model |
| **Architecture** | **Modular** | Monolithic | Basic |

---

##  📂 Project Structure

```
shape/
├── configs/                    # 配置文件
│   ├── config.py              # 配置管理
│   └── deepspeed/             # DeepSpeed 配置
│       ├── zero2.json
│       ├── zero3.json
│       └── zero3_offload.json
├── shape/                      # 核心模块
│   ├── core/                  # 核心工具
│   │   ├── logger.py          # 日志系统
│   │   └── utils.py           # 工具函数
│   ├── training/              # 训练模块
│   │   ├── preference_trainer.py  # 偏好对齐训练器
│   │   ├── llava_dpo_trainer.py   # DPO 训练器
│   │   └── base_dpo_trainer.py    # 基础训练器
│   └── evaluation/            # 评估模块
│       ├── hallucination_metrics.py  # 幻觉指标
│       └── benchmarks.py      # 基准测试
├── src/                        # LLaVA 源码
│   └── llava/                 # LLaVA 模型实现
├── datasets/                   # 训练数据
│   ├── ocrvqa_answer_file_8k_dpo.jsonl
│   └── textvqa_answer_file_8k_dpo.jsonl
├── train_preference_alignment.py  # 训练主脚本
├── fused_inference.py         # 融合推理脚本
├── train.sh                   # 训练启动脚本
├── pyproject.toml             # 项目配置
└── README.md                  # 本文档
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# 创建虚拟环境
conda create -n shape python=3.10 -y
conda activate shape

# 安装依赖
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .

# 安装训练依赖
pip install -e ".[training]"
```

### Data Preparation

#### 1. Preference Data

We provide pre-generated DPO training data in the datasets/ folder:

```
datasets/
├── ocrvqa_answer_file_8k_dpo.jsonl      # OCR-VQA 偏好对
└── textvqa_answer_file_8k_dpo.jsonl     # TextVQA 偏好对
```

Data Format:
```json
{
  "chosen": "正确或更好的回答",
  "reject": "错误或较差的回答",
  "question": "问题文本",
  "image_id": "图像文件名"
}
```

#### 2. Image Data

Please download the corresponding image datasets and place them in the data/ directory:

```bash
# 创建数据目录
mkdir -p data/textvqa data/ocrvqa

# 下载 TextVQA 图像
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip -d data/textvqa/

# 下载 OCR-VQA 图像（参考官方说明）
# https://ocr-vqa.github.io/
```

### Model Preparation

Clone the base model and reward model weights:

```bash
# 基础模型
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

# 奖励模型
git clone https://huggingface.co/bczhou/tiny-llava-v1-hf
```

---

##  Training

### Quick Run
You can start training using the provided shell script:

```bash
bash train.sh
```

### Custom Training Command

编辑 `train.sh` 或直接运行：

```bash
python train_preference_alignment.py \
    --model_name_or_path bczhou/tiny-llava-v1-hf \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --ocr_data_path ./datasets/ocrvqa_answer_file_8k_dpo.jsonl \
    --ocr_image_path data/ocrvqa/images/ \
    --textvqa_data_path ./datasets/textvqa_answer_file_8k_dpo.jsonl \
    --textvqa_image_path data/textvqa/train_images \
    --output_dir checkpoints/shape_model \
    --beta 0.1 \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed configs/deepspeed/zero3_offload.json
```

### Key Arguments:

| Argument | Description | Default |
|-----|------|--------|
| `--beta` | DPO loss 的温度参数 | 0.1 |
| `--learning_rate` | 学习率 | 2e-6 |
| `--num_train_epochs` | 训练轮数 | 1 |
| `--per_device_train_batch_size` | 每个设备的批次大小 | 8 |
| `--gradient_checkpointing` | 梯度检查点（节省显存） | True |
| `--bf16` | 使用 BF16 混合精度 | True |

---

## 🔮 Inference

### Fused Inference

Perform inference by fusing logits from the Base Model and the Reward Model:

```bash
python fused_inference.py \
    --base-model path/to/llava-1.5-7b \
    --reward-model path/to/tiny-llava \
    --weight-base 0.7 \
    --weight-reward 0.3 \
    --dataset MM-Vet/mm-vet \
    --output results/fused_inference.json
```


### Other Benchmarks

For other comprehensive benchmarks (e.g., MME, MMBench), please refer to the official LLaVA Evaluation Docs.



## 🙏 Acknowledgements

This project is built upon the following excellent works:

- [**LLaVA**](https://github.com/haotian-liu/LLaVA) - 大型视觉-语言模型
- [**DPO**](https://arxiv.org/abs/2305.18290) - 直接偏好优化方法
- [**SeVa**](https://github.com/Kevinz-code/SeVa) - 自监督视觉偏好对齐
- [**HA-DPO**](https://github.com/opendatalab/HA-DPO/) - 幻觉感知的 DPO

---

## 📝 Citation

如果您使用本项目，请引用：

```bibtex
@article{chen2026self,
  title={Self-Improved Holistic Alignment for Preference Enhancement},
  author={Chen, Kejia and Zhang, Jiawen and Yang, Jiazhen and Song, Mingli and Feng, Zunlei},
  journal={Pattern Recognition},
  pages={113238},
  year={2026},
  publisher={Elsevier}
}
```

---

##  许可证

This project is licensed under the Apache License 2.0.

---

<div align="center">

⭐ If this project helps you, please give us a Star! ⭐

Made with ❤️ by SHAPE Team

</div>
