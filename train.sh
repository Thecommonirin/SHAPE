#!/bin/bash
# SHAPE Training Script
# 偏好对齐训练启动脚本

set -e

# 配置
MODEL_VERSION="shape_model"
DEEPSPEED_CONFIG="configs/deepspeed/zero3_offload.json"

# 数据路径
OCR_DPO_DATA="./datasets/ocrvqa_answer_file_8k_dpo.jsonl"
TEXT_DPO_DATA="./datasets/textvqa_answer_file_8k_dpo.jsonl"
OCR_IMAGE_PATH="data/ocrvqa/images/"
TEXTVQA_IMAGE_PATH="data/textvqa/train_images"

# 模型配置
MODEL_NAME_OR_PATH="bczhou/tiny-llava-v1-hf"
VISION_TOWER="openai/clip-vit-large-patch14-336"

# 训练配置
BETA=0.1
LEARNING_RATE=2e-6
NUM_EPOCHS=1
BATCH_SIZE=8
GRADIENT_ACCUM_STEPS=1

# GPU 配置（修改为你的 GPU 配置）
GPU_IDS="0,1,2,3"

# 输出目录
OUTPUT_DIR="checkpoints/${MODEL_VERSION}"
mkdir -p ${OUTPUT_DIR}

echo "================================"
echo "SHAPE 偏好对齐训练"
echo "================================"
echo "模型: ${MODEL_NAME_OR_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "Beta: ${BETA}"
echo "学习率: ${LEARNING_RATE}"
echo "训练轮数: ${NUM_EPOCHS}"
echo "================================"

# 启动训练
deepspeed --include localhost:${GPU_IDS} train_preference_alignment.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --version v1 \
    --ocr_data_path ${OCR_DPO_DATA} \
    --ocr_image_path ${OCR_IMAGE_PATH} \
    --textvqa_data_path ${TEXT_DPO_DATA} \
    --textvqa_image_path ${TEXTVQA_IMAGE_PATH} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCUM_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${MODEL_VERSION} \
    --beta ${BETA}

echo "================================"
echo "训练完成!"
echo "模型保存在: ${OUTPUT_DIR}"
echo "================================"
