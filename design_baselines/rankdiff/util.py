"""Utilities for RankDiff implementation."""

import os
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

# 任务名称到任务ID的映射
TASKNAME2TASK = {
    'dkitty': 'DKittyMorphology-Exact-v0',
    'ant': 'AntMorphology-Exact-v0',
    'tf-bind-8': 'TFBind8-Exact-v0',
    'tf-bind-10': 'TFBind10-Exact-v0',
    'superconductor': 'Superconductor-RandomForest-v0',
    'nas': 'CIFARNAS-Exact-v0',
    'chembl': 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0',
}

def configure_gpu(use_gpu: bool, which_gpu: int) -> torch.device:
    """配置GPU设备"""
    if use_gpu:
        device = torch.device("cuda")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return device

def set_seed(seed: Optional[int]) -> None:
    """设置随机种子"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    metrics = {}
    
    # 计算MSE
    metrics['mse'] = np.mean((predictions - targets) ** 2)
    
    # 计算MAE
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    # 计算R2分数
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    
    return metrics

def normalize_data(data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """标准化数据"""
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    normalized = (data - mean) / (std + 1e-8)
    return normalized, mean, std

def denormalize_data(normalized: np.ndarray, mean: float, std: float) -> np.ndarray:
    """反标准化数据"""
    return normalized * std + mean

def save_results(results: Dict, save_path: str) -> None:
    """保存结果"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, results)

def load_results(load_path: str) -> Dict:
    """加载结果"""
    return np.load(load_path, allow_pickle=True).item()

def get_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """获取优化器"""
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif config['optimizer'] == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    """获取学习率调度器"""
    if config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 0.0)
        )
    elif config['scheduler'] == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['num_epochs']
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}") 