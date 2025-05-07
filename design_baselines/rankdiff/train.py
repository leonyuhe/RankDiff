import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import configargparse
import os
import json
from tqdm import tqdm

from .model import RankDiffModel
from .trainer import RankDiffTrainer

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_dataset(data_path):
    # 加载数据集
    data = torch.load(data_path)
    return {
        'x': data['x'],
        'y': data['y']
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    dataset = load_dataset(args.data_path)
    
    # 创建模型
    model = RankDiffModel(
        input_dim=config['model']['input_dim'],
        time_dim=config['model']['time_dim'],
        condition_dim=config['model']['condition_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_blocks=config['model']['num_blocks']
    ).to(args.device)
    
    # 创建训练器
    trainer = RankDiffTrainer(model, dataset, config['training'])
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 创建数据加载器
    dataset = TensorDataset(dataset['x'], dataset['y'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # 训练循环
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}'):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            
            # 前向传播
            loss_dict = trainer.train_step({'x': x, 'y': y})
            loss = loss_dict['loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))

if __name__ == '__main__':
    main() 