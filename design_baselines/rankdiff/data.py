import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import json

class RankDiffDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        transform: Optional[callable] = None
    ):
        self.x = torch.from_numpy(data['x']).float()
        self.y = torch.from_numpy(data['y']).float()
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

def load_dataset(
    data_path: str,
    task_name: str,
    split: str = 'train'
) -> Dict[str, np.ndarray]:
    """加载数据集"""
    if task_name in ['dkitty', 'ant', 'tf-bind-8', 'tf-bind-10', 'superconductor', 'nas', 'chembl']:
        # 使用design-bench数据集
        from design_bench import make
        task = make(TASKNAME2TASK[task_name])
        
        if split == 'train':
            x = task.x
            y = task.y
        else:
            x = task.x_test
            y = task.y_test
    else:
        # 加载自定义数据集
        data = np.load(data_path, allow_pickle=True).item()
        x = data['x']
        y = data['y']
    
    return {
        'x': x,
        'y': y
    }

def preprocess_data(
    data: Dict[str, np.ndarray],
    normalize: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """预处理数据"""
    stats = {}
    
    if normalize:
        # 标准化x
        x_mean = np.mean(data['x'], axis=0)
        x_std = np.std(data['x'], axis=0)
        data['x'] = (data['x'] - x_mean) / (x_std + 1e-8)
        stats['x_mean'] = x_mean
        stats['x_std'] = x_std
        
        # 标准化y
        y_mean = np.mean(data['y'])
        y_std = np.std(data['y'])
        data['y'] = (data['y'] - y_mean) / (y_std + 1e-8)
        stats['y_mean'] = y_mean
        stats['y_std'] = y_std
    
    return data, stats

def create_dataloader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器"""
    dataset = RankDiffDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def save_preprocessed_data(
    data: Dict[str, np.ndarray],
    stats: Dict[str, float],
    save_dir: str
) -> None:
    """保存预处理后的数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存数据
    np.save(os.path.join(save_dir, 'data.npy'), data)
    
    # 保存统计信息
    with open(os.path.join(save_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)

def load_preprocessed_data(load_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """加载预处理后的数据"""
    # 加载数据
    data = np.load(os.path.join(load_dir, 'data.npy'), allow_pickle=True).item()
    
    # 加载统计信息
    with open(os.path.join(load_dir, 'stats.json'), 'r') as f:
        stats = json.load(f)
    
    return data, stats 