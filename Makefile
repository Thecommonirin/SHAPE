# SHAPE Makefile
# 便捷的命令集合

.PHONY: help install install-dev clean train infer eval format lint test

help:
	@echo "SHAPE - Self-supervised Hallucination Alignment with Preference Enhancement"
	@echo ""
	@echo "可用命令:"
	@echo "  make install      - 安装项目依赖"
	@echo "  make install-dev  - 安装开发依赖"
	@echo "  make clean        - 清理缓存和临时文件"
	@echo "  make train        - 启动训练"
	@echo "  make infer        - 运行推理"
	@echo "  make eval         - 运行评估"
	@echo "  make format       - 格式化代码"
	@echo "  make lint         - 代码检查"
	@echo "  make test         - 运行测试"

install:
	@echo "安装 SHAPE..."
	pip install -e .

install-dev:
	@echo "安装开发依赖..."
	pip install -e ".[dev,training,serving]"

install-training:
	@echo "安装训练依赖..."
	pip install -e ".[training]"

clean:
	@echo "清理缓存文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "清理完成!"

train:
	@echo "启动训练..."
	bash train.sh

infer:
	@echo "运行推理..."
	@echo "请设置模型路径后运行: python fused_inference.py --base-model <path> --reward-model <path>"

eval:
	@echo "运行评估..."
	@echo "请参考 README.md 中的评估说明"

format:
	@echo "格式化代码..."
	black . --line-length 120
	isort . --profile black

lint:
	@echo "代码检查..."
	flake8 shape/ --max-line-length 120 --ignore E203,W503
	mypy shape/ --ignore-missing-imports

test:
	@echo "运行测试..."
	pytest tests/ -v

check: format lint test
	@echo "代码检查完成!"

.DEFAULT_GOAL := help
