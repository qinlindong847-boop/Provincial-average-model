"""
TDNN 电价价差预测训练脚本（不含天气特征）
用于对比验证天气特征的影响
"""

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


# ===================== 辅助函数 =====================
def calculate_direction_accuracy(y_true, y_pred):
    """计算方向准确率：预测的涨跌方向与实际是否一致"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    mask = (true_diff != 0)
    if np.sum(mask) == 0:
        return 0.0
    
    correct = np.sum((true_diff[mask] * pred_diff[mask]) > 0)
    return correct / np.sum(mask)


def calculate_mape(y_true, y_pred):
    """计算 MAPE（平均绝对百分比误差）"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) == 0:
        return float('inf')
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


# ===================== 1. 配置 =====================
class Config:
    # ===== 数据参数 =====
    province = '广东'
    start_date = '2024-01-01'
    end_date = '2026-6-30'
    
    # ===== 时间窗口 =====
    input_days = 7
    output_days = 1
    points_per_day = 96
    input_len = input_days * points_per_day      # 672
    output_len = output_days * points_per_day    # 96
    
    # ===== 特征列（不含天气） =====
    feature_cols = [
        'ahead_price_data',      # 日前电价（全省平均）
        'real_price_data',       # 实时电价（全省平均）
        'price_spread',    
        'block_weight',
        'maintenance_weight',
        'is_workday'             # 是否工作日
    ]
    
    feature_dim = len(feature_cols)
    
    # ===== 训练参数 =====
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
        feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        # MinMax 归一化
        self.feature_min = feature_data.min(axis=0, keepdims=True)
        self.feature_max = feature_data.max(axis=0, keepdims=True)
        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range < 1e-8] = 1.0
        feature_data = (feature_data - self.feature_min) / self.feature_range
        
        # 目标：价差
        price_spread = df['price_spread'].values.astype(np.float32)
        price_spread = np.clip(price_spread, -300, 300)
        price_spread = np.nan_to_num(price_spread, nan=0.0)
        
        self.target_min = price_spread.min()
        self.target_max = price_spread.max()
        self.target_range = self.target_max - self.target_min
        if self.target_range < 1e-8:
            self.target_range = 1.0
        price_spread = (price_spread - self.target_min) / self.target_range
        
        # 滑动窗口生成样本
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
        
        # 输入维度变为 4（少了天气特征）
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
    print("TDNN 电价价差预测训练（不含天气特征）")
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
        end_date=config.end_date
    )
    print(f"数据行数: {len(df):,}")
    
    # 创建数据集
    dataset_obj = PriceSpreadDataset(df, config)
    print(f"样本数: {len(dataset_obj):,}")
    
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
    
    print(f"训练集: {len(train_dataset):,} 样本")
    print(f"验证集: {len(val_dataset):,} 样本")
    
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
            torch.save(model.state_dict(), 'tdnn_best_model_no_weather.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n⏹️ 早停触发！Epoch {epoch+1}")
                break
    
    print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('tdnn_best_model_no_weather.pt'))
    model.eval()
    
    # 评估
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
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    dir_acc = calculate_direction_accuracy(all_targets, all_preds)
    mape = calculate_mape(all_targets, all_preds)
    
    print("\n" + "=" * 60)
    print("📊 验证集评估（不含天气）")
    print("=" * 60)
    print(f"整体 RMSE: {rmse:.4f} 元/MWh")
    print(f"整体 MAE : {mae:.4f} 元/MWh")
    print(f"整体 R²  : {r2:.4f}")
    print(f"方向准确率: {dir_acc:.2%}")
    if mape != float('inf'):
        print(f"整体 MAPE : {mape:.2f}%")
    
    # 保存结果
    eval_results = {
        'version': 'no_weather',
        'feature_cols': config.feature_cols,
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'direction_accuracy': float(dir_acc),
        'mape': float(mape) if mape != float('inf') else None
    }
    with open('tdnn_eval_results_no_weather.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n💾 模型已保存: tdnn_best_model_no_weather.pt")
    print("💾 评估结果已保存: tdnn_eval_results_no_weather.json")
    
    return model, dataset_obj


if __name__ == '__main__':
    train()