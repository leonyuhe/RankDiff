import os
import argparse
import json
import torch
from design_baselines.rankdiff.data import load_dataset, preprocess_data, save_preprocessed_data
from design_baselines.rankdiff.train import main as train_main
from design_baselines.rankdiff.eval import main as eval_main
from design_baselines.rankdiff.util import configure_gpu, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both',
                      help='Run mode: train, eval, or both')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 配置GPU
    device = configure_gpu(True, args.gpu)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据预处理
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 加载数据集
    data = load_dataset(
        data_path=None,  # 使用design-bench数据集
        task_name=config['data']['task_name'],
        split='train'
    )
    
    # 预处理数据
    data, stats = preprocess_data(
        data,
        normalize=config['data']['normalize']
    )
    
    # 保存预处理后的数据
    save_preprocessed_data(data, stats, data_dir)
    
    # 训练模型
    if args.mode in ['train', 'both']:
        train_args = argparse.Namespace(
            config=args.config,
            data_path=os.path.join(data_dir, 'data.npy'),
            output_dir=os.path.join(args.output_dir, 'checkpoints'),
            device=device
        )
        train_main(train_args)
    
    # 评估模型
    if args.mode in ['eval', 'both']:
        eval_args = argparse.Namespace(
            config=args.config,
            model_path=os.path.join(args.output_dir, 'checkpoints', 'final_model.pt'),
            data_path=os.path.join(data_dir, 'data.npy'),
            output_dir=os.path.join(args.output_dir, 'results'),
            device=device
        )
        eval_main(eval_args)

if __name__ == '__main__':
    main() 