"""
TDNN 模型回测脚本
用历史数据模拟预测，评估模型在连续时间段内的真实表现
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from dataset import dataset


# ===================== 1. 配置 =====================
class Config:
    province = '广东'
    start_date = '2024-01-01'
    end_date = '2026-06-30'
    
    input_days = 7
    output_days = 1
    points_per_day = 96
    input_len = input_days * points_per_day
    output_len = output_days * points_per_day
    
    feature_cols = [
        'ahead_price_data',
        'real_price_data',
        'price_spread',
        'is_workday'
    ]
    feature_dim = len(feature_cols)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = 128


# ===================== 2. 数据集 =====================
class PriceSpreadDataset:
    def __init__(self, df, config):
        self.config = config
        self.feature_cols = config.feature_cols
        self.input_len = config.input_len
        self.output_len = config.output_len
        self._prepare_data(df)
        
    def _prepare_data(self, df):
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        feature_data = df[self.feature_cols].values.astype(np.float32)
        feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        self.feature_min = feature_data.min(axis=0, keepdims=True)
        self.feature_max = feature_data.max(axis=0, keepdims=True)
        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range < 1e-8] = 1.0
        
        price_spread = df['price_spread'].values.astype(np.float32)
        price_spread = np.clip(price_spread, -500, 500)
        price_spread = np.nan_to_num(price_spread, nan=0.0)
        
        self.target_min = price_spread.min()
        self.target_max = price_spread.max()
        self.target_range = self.target_max - self.target_min
        if self.target_range < 1e-8:
            self.target_range = 1.0


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


# ===================== 4. 辅助函数 =====================
def calculate_direction_accuracy(y_true, y_pred):
    """计算方向准确率"""
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
    """计算 MAPE"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) == 0:
        return float('inf')
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


# ===================== 5. 回测函数 =====================
def backtest_month(year, month, model_path='tdnn_best_model_no_weather.pt'):
    """
    对指定月份进行回测
    year: 年份，如 2026
    month: 月份，如 7
    """
    config = Config()
    
    print("=" * 60)
    print(f"📊 TDNN 回测: {year}年{month}月")
    print("=" * 60)
    
    # ===== 1. 加载模型 =====
    model = TDNN(config)
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    print(f"✅ 模型加载成功")
    
    # ===== 2. 准备回测数据 =====
    # 回测月份的前后各多取10天（用于构造7天窗口）
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-01"
    
    # 计算月份天数
    if month == 12:
        next_month = f"{year+1}-01-01"
    else:
        next_month = f"{year}-{month+1:02d}-01"
    
    from datetime import datetime as dt
    month_start = dt(year, month, 1)
    if month == 12:
        month_end = dt(year+1, 1, 1) - timedelta(days=1)
    else:
        month_end = dt(year, month+1, 1) - timedelta(days=1)
    
    days_in_month = month_end.day
    
    # 数据范围：月初往前推10天，到月底
    data_start = (month_start - timedelta(days=10)).strftime('%Y-%m-%d')
    data_end = month_end.strftime('%Y-%m-%d')
    
    print(f"\n📥 加载数据...")
    print(f"   数据范围: {data_start} ~ {data_end}")
    
    df = dataset(
        province=config.province,
        start_date=data_start,
        end_date=data_end
    )
    
    if len(df) == 0:
        print("❌ 没有获取到数据")
        return None
    
    print(f"   数据行数: {len(df)}")
    
    # ===== 3. 获取归一化参数 =====
    dataset_obj = PriceSpreadDataset(df, config)
    
    # ===== 4. 逐日滚动预测 =====
    print(f"\n🔮 逐日预测 {days_in_month} 天...")
    
    all_preds = []
    all_targets = []
    all_dates = []
    
    # 按天分组，找到每个日期的96点数据
    df['date'] = df['timestamp'].dt.date
    daily_data = df.groupby('date')
    
    # 获取每天的数据
    daily_dict = {}
    for date, group in daily_data:
        if len(group) == 96:
            daily_dict[date] = group.sort_values('timestamp')
    
    # 按日期排序
    sorted_dates = sorted(daily_dict.keys())
    
    # 只回测目标月份
    target_dates = [d for d in sorted_dates if d.year == year and d.month == month]
    
    print(f"   目标月份有 {len(target_dates)} 天数据")
    
    if len(target_dates) < 10:
        print("⚠️ 数据不足，至少需要10天才能回测")
        return None
    
    for i, target_date in enumerate(target_dates):
        # 需要前7天的数据
        date_idx = sorted_dates.index(target_date)
        if date_idx < 7:
            continue
        
        # 获取过去7天的数据
        past_dates = sorted_dates[date_idx-7:date_idx]
        
        # 提取过去7天的特征
        past_features = []
        for d in past_dates:
            day_data = daily_dict[d]
            past_features.append(day_data[config.feature_cols].values)
        
        X_input = np.concatenate(past_features, axis=0)  # (672, 4)
        
        # 归一化
        X_input = (X_input - dataset_obj.feature_min) / dataset_obj.feature_range
        X_tensor = torch.FloatTensor(X_input).unsqueeze(0).to(config.device)
        
        # 预测
        with torch.no_grad():
            pred_norm = model(X_tensor).cpu().numpy()[0]
        
        pred = pred_norm * dataset_obj.target_range + dataset_obj.target_min
        
        # 真实值
        target = daily_dict[target_date]['price_spread'].values
        
        all_preds.append(pred)
        all_targets.append(target)
        all_dates.append(target_date)
    
    if len(all_preds) == 0:
        print("❌ 没有有效预测")
        return None
    
    # ===== 5. 计算指标 =====
    print("\n" + "=" * 60)
    print(f"📊 回测结果: {year}年{month}月")
    print("=" * 60)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 整体指标
    rmse = np.sqrt(np.mean((all_targets.flatten() - all_preds.flatten()) ** 2))
    mae = np.mean(np.abs(all_targets.flatten() - all_preds.flatten()))
    
    # 计算 R²
    y_true_flat = all_targets.flatten()
    y_pred_flat = all_preds.flatten()
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # 方向准确率
    dir_acc = calculate_direction_accuracy(all_targets, all_preds)
    
    # MAPE
    mape = calculate_mape(all_targets, all_preds)
    
    print(f"📅 预测天数: {len(all_preds)} 天")
    print(f"📊 整体 R²  : {r2:.4f}")
    print(f"📊 RMSE: {rmse:.2f} 元/MWh")
    print(f"📊 MAE : {mae:.2f} 元/MWh")
    print(f"📊 方向准确率: {dir_acc:.2%}")
    if mape != float('inf'):
        print(f"📊 MAPE : {mape:.2f}%")
    
    # ===== 6. 按天统计 =====
    daily_errors = []
    for i in range(len(all_preds)):
        day_rmse = np.sqrt(np.mean((all_targets[i] - all_preds[i]) ** 2))
        day_mae = np.mean(np.abs(all_targets[i] - all_preds[i]))
        day_dir = calculate_direction_accuracy(all_targets[i], all_preds[i])
        daily_errors.append({
            'date': all_dates[i],
            'rmse': day_rmse,
            'mae': day_mae,
            'direction_accuracy': day_dir
        })
    
    daily_df = pd.DataFrame(daily_errors)
    
    print(f"\n📊 每日统计:")
    print(f"   平均 RMSE: {daily_df['rmse'].mean():.2f} 元/MWh")
    print(f"   平均 MAE : {daily_df['mae'].mean():.2f} 元/MWh")
    print(f"   平均方向准确率: {daily_df['direction_accuracy'].mean():.2%}")
    
    # ===== 7. 绘图 =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 7.1 预测 vs 真实（取中间一天展示）
    mid_idx = len(all_preds) // 2
    axes[0].plot(all_targets[mid_idx], label='真实值', linewidth=2, color='blue')
    axes[0].plot(all_preds[mid_idx], label='预测值', linewidth=2, color='orange', alpha=0.8)
    axes[0].set_title(f'{all_dates[mid_idx].strftime("%Y-%m-%d")} 预测 vs 真实')
    axes[0].set_xlabel('时间点 (15分钟)')
    axes[0].set_ylabel('price_spread (元/MWh)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 7.2 每日方向准确率
    axes[1].bar(range(len(daily_df)), daily_df['direction_accuracy'], color='skyblue')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='随机猜测 (50%)')
    axes[1].set_title(f'每日方向准确率 (月均: {daily_df["direction_accuracy"].mean():.2%})')
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('方向准确率')
    axes[1].set_xticks(range(len(daily_df)))
    axes[1].set_xticklabels([d.strftime('%m-%d') for d in all_dates], rotation=45)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    output_png = f'tdnn_backtest_{year}_{month:02d}.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n📊 回测图已保存: {output_png}")
    plt.show()
    



# ===================== 6. 主程序 =====================
if __name__ == '__main__':
    import sys
    
    # 默认回测 2026 年 6 月
    year = 2026
    month = 6
    
    # 支持命令行参数：python tdnn_backtest.py 2026 7
    if len(sys.argv) >= 3:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
    
    backtest_month(year, month)