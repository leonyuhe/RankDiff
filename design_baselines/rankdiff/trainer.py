import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F

class RankDiffTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: Dict[str, torch.Tensor],
        config: Dict
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # 计算数据集统计信息
        self.y_mean = torch.mean(dataset['y'])
        self.y_std = torch.std(dataset['y'])
        
        # 计算z-score归一化的y值
        self.y_zscore = (dataset['y'] - self.y_mean) / self.y_std
        
        # 计算排名和权重
        self.ranks = torch.argsort(torch.argsort(self.y_zscore))
        self.rank_weights = 1 + self.config['beta_w'] * (self.ranks / len(self.ranks))
        
        # 设置噪声调度
        self.base_noise_schedule = torch.linspace(
            self.config['beta_start'],
            self.config['beta_end'],
            self.config['num_timesteps']
        )
        
    def get_dynamic_noise_schedule(self, y_zscore: torch.Tensor) -> torch.Tensor:
        """计算动态噪声调度"""
        y_min = torch.min(self.y_zscore)
        y_max = torch.max(self.y_zscore)
        y_normalized = (y_zscore - y_min) / (y_max - y_min)
        
        # 根据公式(5)计算动态噪声调度
        noise_adjustment = 1 - self.config['alpha'] * y_normalized
        return self.base_noise_schedule * noise_adjustment.view(-1, 1)
    
    def forward_diffusion(self, x0: torch.Tensor, y_zscore: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散过程"""
        beta_t = self.get_dynamic_noise_schedule(y_zscore)
        alpha_t = 1 - beta_t
        
        # 生成噪声
        noise = torch.randn_like(x0)
        
        # 计算xt
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(beta_t) * noise
        
        return xt, noise
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """单步训练"""
        x0 = batch['x']
        y = batch['y']
        y_zscore = (y - self.y_mean) / self.y_std
        
        # 随机采样时间步
        t = torch.randint(0, self.config['num_timesteps'], (x0.shape[0],))
        
        # 前向扩散
        xt, noise = self.forward_diffusion(x0, y_zscore, t)
        
        # 计算模型预测
        pred_noise = self.model(xt, t, y_zscore)
        
        # 计算损失权重
        batch_ranks = torch.argsort(torch.argsort(y_zscore))
        weights = 1 + self.config['beta_w'] * (batch_ranks / len(batch_ranks))
        
        # 计算损失
        loss = weights * F.mse_loss(pred_noise, noise, reduction='none')
        loss = loss.mean()
        
        return {'loss': loss}
    
    def sample(self, y_target: float, num_samples: int = 256) -> torch.Tensor:
        """生成样本"""
        # 计算目标y的z-score
        y_target_zscore = (y_target - self.y_mean) / self.y_std
        
        # 初始化噪声
        x = torch.randn(num_samples, *self.config['input_shape'])
        
        # 使用Heun采样器进行反向扩散
        for t in reversed(range(self.config['num_timesteps'])):
            t_batch = torch.full((num_samples,), t)
            
            # 计算条件分数
            eps_cond = self.model(x, t_batch, y_target_zscore)
            
            # 计算无条件分数（用于CFG）
            eps_uncond = self.model(x, t_batch, torch.zeros_like(y_target_zscore))
            
            # 应用分类器自由引导
            eps = (1 + self.config['gamma']) * eps_cond - self.config['gamma'] * eps_uncond
            
            # 更新x
            alpha_t = 1 - self.base_noise_schedule[t]
            x = (x - torch.sqrt(self.base_noise_schedule[t]) * eps) / torch.sqrt(alpha_t)
            
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(self.base_noise_schedule[t]) * noise
        
        return x 