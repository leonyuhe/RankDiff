# RankDiff: Rank-Based Diffusion Models for Offline Black-Box Optimization

This repository contains the implementation of RankDiff, a novel approach for offline black-box optimization using conditional diffusion models with rank-based reweighting and dynamic noise scheduling.d

## Features

- Dynamic Noise Scheduling: Adjusts noise rates based on function values
- Rank-Based Reweighting: Simplified loss weighting using normalized ranks
- Conditional Training with Classifier-Free Guidance
- Efficient sampling using Heun solver

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RankDiff.git
cd RankDiff
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
RankDiff/
├── configs/                    # 配置文件目录
│   ├── rankdiff.cfg           # 基础配置文件
│   └── experiments/           # 实验配置文件
│       └── dkitty.cfg         # DKitty任务配置
├── design_baselines/          # 主要代码目录
│   └── rankdiff/             # RankDiff实现
│       ├── __init__.py
│       ├── model.py          # 模型定义
│       ├── trainer.py        # 训练器
│       ├── data.py           # 数据处理
│       ├── eval.py           # 评估脚本
│       └── util.py           # 工具函数
├── run_experiment.py          # 实验运行脚本
└── README.md
```

## Quick Start

### Running Experiment

Use the `run_experiment.py` script to run an experiment:

```bash
python run_experiment.py \
    --config configs/experiments/dkitty.cfg \
    --output_dir experiments/dkitty \
    --seed 42 \
    --gpu 0 \
    --mode both
```

Parameters:
- `--config`: Path to the experiment configuration file
- `--output_dir`: Output directory
- `--seed`: Random seed
- `--gpu`: GPU ID
- `--mode`: Run mode, options are 'train', 'eval', or 'both'

### Data Preprocessing

```python
from design_baselines.rankdiff.data import load_dataset, preprocess_data

# Load data
data = load_dataset(data_path, task_name)

# Preprocess data
data, stats = preprocess_data(data, normalize=True)
```

### Model Training

```bash
python -m design_baselines.rankdiff.train \
    --config configs/rankdiff.cfg \
    --data_path path/to/your/dataset.pt \
    --output_dir path/to/output \
    --device cuda
```

### Model Evaluation

```bash
python -m design_baselines.rankdiff.eval \
    --config configs/rankdiff.cfg \
    --model_path path/to/model.pt \
    --data_path path/to/test_data.pt \
    --output_dir path/to/output \
    --device cuda
```

## Configuration

### Model Parameters
- `input_dim`: Input dimension
- `time_dim`: Time embedding dimension
- `condition_dim`: Condition embedding dimension
- `hidden_dim`: Hidden layer dimension
- `num_blocks`: Number of Transformer blocks

### Training Parameters
- `num_timesteps`: Number of diffusion steps
- `beta_start`: Noise schedule start value
- `beta_end`: Noise schedule end value
- `alpha`: Ranking loss weight
- `beta_w`: Target function weight
- `gamma`: Condition guidance weight
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `optimizer`: Optimizer type
- `weight_decay`: Weight decay
- `scheduler`: Learning rate scheduler
- `min_lr`: Minimum learning rate

### Sampling Parameters
- `num_samples`: Number of samples
- `guidance_scale`: Condition guidance scale

### Data Parameters
- `task_name`: Task name
- `normalize`: Whether to normalize data

## Experiment Results

Experiment results will be saved in the specified output directory, including:
- Preprocessed data
- Model checkpoints
- Evaluation results and visualizations

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{rankdiff2024,
  title={RankDiff: A Ranking-based Diffusion Model for Design Optimization},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
