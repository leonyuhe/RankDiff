import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 使用正弦位置编码
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return self.projection(embeddings)

class ConditionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.projection(y.unsqueeze(-1))

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        
        self.time_proj = nn.Linear(time_dim, dim)
        self.condition_proj = nn.Linear(condition_dim, dim)
        
        self.norm2 = nn.GroupNorm(8, dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        
        self.residual_conv = nn.Conv2d(dim, dim, 1) if dim != dim else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, condition_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        h = h + self.time_proj(time_emb)[:, :, None, None]
        # 添加条件嵌入
        h = h + self.condition_proj(condition_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        return h + self.residual_conv(x)

class RankDiffModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int = 256,
        condition_dim: int = 256,
        hidden_dim: int = 256,
        num_blocks: int = 4
    ):
        super().__init__()
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_dim)
        
        # 条件嵌入
        self.condition_embedding = ConditionEmbedding(condition_dim)
        
        # 初始投影
        self.init_conv = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        
        # 残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim, condition_dim)
            for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        # 计算嵌入
        time_emb = self.time_embedding(t)
        condition_emb = self.condition_embedding(y)
        
        # 初始特征
        h = self.init_conv(x)
        
        # 通过残差块
        for block in self.blocks:
            h = block(h, time_emb, condition_emb)
        
        # 输出
        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        
        return h 