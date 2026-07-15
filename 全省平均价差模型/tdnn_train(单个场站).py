import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from dataset import dataset


# ===================== 1. 配置 =====================
class Config:
    province = '广东'
    classify_id = 139
    station_name = '星子光伏电站'
    start_date = '2024-01-01'
    end_date = '2025-12-31'
    
    input_days = 7
    output_days = 1
    points_per_day = 96
    input_len = input_days * points_per_day
    output_len = output_days * points_per_day
    
    feature_cols = [
        'ahead_price_data', 'real_price_data', 'price_spread',
        'temperature_2m', 'precipitation', 'wind_speed_100m',
        'sunshine_duration', 'shortwave_radiation',
        'is_workday'
    ]
    feature_dim = len(feature_cols)
    
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 200
    patience = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = 128


# ===================== 2. 数据集 =====================
class PriceSpreadDataset(Dataset):
    def __init__(self, df, config):
        self.config = config
        self.feature_cols = config.feature_cols
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.data = self._prepare_data(df)
        
    def _prepare_data(self, df):
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 提取特征
        feature_data = df[self.feature_cols].values.astype(np.float32)
        
        # 处理 NaN（用 0 填充）
        feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        # MinMax 归一化
        self.feature_min = feature_data.min(axis=0, keepdims=True)
        self.feature_max = feature_data.max(axis=0, keepdims=True)
        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range < 1e-8] = 1.0
        feature_data = (feature_data - self.feature_min) / self.feature_range
        
        # 目标：价差（截断异常值）
        price_spread = df['price_spread'].values.astype(np.float32)
        price_spread = np.clip(price_spread, -500, 500)  # 截断异常值
        price_spread = np.nan_to_num(price_spread, nan=0.0)
        
        self.target_min = price_spread.min()
        self.target_max = price_spread.max()
        self.target_range = self.target_max - self.target_min
        if self.target_range < 1e-8:
            self.target_range = 1.0
        price_spread = (price_spread - self.target_min) / self.target_range
        
        # 生成样本
        samples = []
        total_points = len(df)
        for i in range(self.input_len, total_points - self.output_len + 1):
            X = feature_data[i - self.input_len:i]
            y = price_spread[i:i + self.output_len]
            samples.append((X, y))
        
        return samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, y = self.data[idx]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ===================== 3. TDNN模型 =====================
class TDNN(nn.Module):
    def __init__(self, config):
        super(TDNN, self).__init__()
        
        self.conv1 = nn.Conv1d(config.feature_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.fc1 = nn.Linear(256 * 32, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.output_len)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===================== 4. 训练函数 =====================
def train():
    config = Config()
    print("=" * 60)
    print("TDNN 电价价差预测训练（修复版）")
    print("=" * 60)
    print(f"设备: {config.device}")
    print(f"特征数: {config.feature_dim}")
    print(f"特征: {config.feature_cols}")
    print("=" * 60)
    
    # 加载数据
    print("\n📥 加载数据...")
    df = dataset(
        province=config.province,
        start_date=config.start_date,
        end_date=config.end_date,
        classify_id=config.classify_id,
        station_name=config.station_name
    )
    print(f"数据行数: {len(df)}")
    
    # 创建数据集
    dataset_obj = PriceSpreadDataset(df, config)
    print(f"样本数: {len(dataset_obj)}")
    
    if len(dataset_obj) == 0:
        print("❌ 没有样本！")
        return None
    
    # 划分数据集
    train_size = int(len(dataset_obj) * 0.8)
    val_size = len(dataset_obj) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset_obj, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 模型
    model = TDNN(config).to(config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 训练
    print("\n" + "=" * 60)
    print("🚀 开始训练...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                output = model(X)
                loss = criterion(output, y)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'tdnn_best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n⏹️ 早停触发！Epoch {epoch+1}")
                break
    
    print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")
    
    # 评估
    model.load_state_dict(torch.load('tdnn_best_model.pt'))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(config.device), y.to(config.device)
            output = model(X)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 反归一化
    all_preds = all_preds * dataset_obj.target_range + dataset_obj.target_min
    all_targets = all_targets * dataset_obj.target_range + dataset_obj.target_min
    
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    
    print("\n" + "=" * 60)
    print("📊 验证集评估")
    print("=" * 60)
    print(f"整体 RMSE: {rmse:.4f} 元/MWh")
    print(f"整体 MAE : {mae:.4f} 元/MWh")
    print(f"整体 R²  : {r2:.4f}")
    
    # 保存结果
    eval_results = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
    with open('tdnn_eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("💾 模型已保存: tdnn_best_model.pt")
    return model, dataset_obj


if __name__ == '__main__':
    train()