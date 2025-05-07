import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from .model import RankDiffModel
from .trainer import RankDiffTrainer
from .util import compute_metrics, save_results, load_results

def evaluate_model(
    model: nn.Module,
    trainer: RankDiffTrainer,
    test_data: Dict[str, torch.Tensor],
    config: Dict,
    device: torch.device
) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    results = {}
    
    with torch.no_grad():
        # 生成样本
        y_target = torch.max(test_data['y']).item()
        samples = trainer.sample(y_target, num_samples=config['sampling']['num_samples'])
        
        # 计算评估指标
        metrics = compute_metrics(
            samples.cpu().numpy(),
            test_data['x'].cpu().numpy()
        )
        results.update(metrics)
        
        # 计算目标函数值
        if 'f' in test_data:
            f_values = test_data['f'](samples)
            results['target_f'] = f_values.mean().item()
            results['best_f'] = f_values.max().item()
    
    return results

def visualize_results(
    results: Dict[str, float],
    save_dir: str
) -> None:
    """可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    if 'train_loss' in results and 'val_loss' in results:
        plt.figure(figsize=(10, 5))
        plt.plot(results['train_loss'], label='Training Loss')
        plt.plot(results['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()
    
    # 绘制性能指标
    metrics = {k: v for k, v in results.items() if k not in ['train_loss', 'val_loss']}
    if metrics:
        plt.figure(figsize=(10, 5))
        plt.bar(metrics.keys(), metrics.values())
        plt.xticks(rotation=45)
        plt.title('Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 加载数据
    test_data = torch.load(args.data_path)
    
    # 创建模型
    model = RankDiffModel(
        input_dim=config['model']['input_dim'],
        time_dim=config['model']['time_dim'],
        condition_dim=config['model']['condition_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_blocks=config['model']['num_blocks']
    ).to(args.device)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建训练器
    trainer = RankDiffTrainer(model, test_data, config['training'])
    
    # 评估模型
    results = evaluate_model(model, trainer, test_data, config, args.device)
    
    # 保存结果
    save_results(results, os.path.join(args.output_dir, 'eval_results.npy'))
    
    # 可视化结果
    visualize_results(results, args.output_dir)
    
    # 打印结果
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 