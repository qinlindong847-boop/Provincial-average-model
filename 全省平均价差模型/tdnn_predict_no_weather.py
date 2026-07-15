"""
TDNN 预测脚本（不含天气）
加载训练好的模型，预测未来1天的96点价差
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
    input_len = input_days * points_per_day      # 672
    output_len = output_days * points_per_day    # 96
    
    # 不含天气的特征（和训练保持一致）
    feature_cols = [
        'ahead_price_data',
        'real_price_data',
        'price_spread',
        'is_workday',
         'block_weight',          # 阻塞权重
         'maintenance_weight'
    ]
    
    feature_dim = len(feature_cols)
    
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = 128


# ===================== 2. 数据集（用于获取归一化参数） =====================
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
        
        # 保存归一化参数
        self.feature_min = feature_data.min(axis=0, keepdims=True)
        self.feature_max = feature_data.max(axis=0, keepdims=True)
        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range < 1e-8] = 1.0
        
        # 目标：价差
        price_spread = df['price_spread'].values.astype(np.float32)
        price_spread = np.clip(price_spread, -300, 300)
        price_spread = np.nan_to_num(price_spread, nan=0.0)
        
        self.target_min = price_spread.min()
        self.target_max = price_spread.max()
        self.target_range = self.target_max - self.target_min
        if self.target_range < 1e-8:
            self.target_range = 1.0


# ===================== 3. TDNN模型（和训练一致） =====================
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


# ===================== 4. 预测函数 =====================
def predict_next_day():
    config = Config()
    print("=" * 60)
    print("TDNN 电价价差预测（不含天气）")
    print("=" * 60)
    print(f"设备: {config.device}")
    print(f"特征: {config.feature_cols}")
    print("=" * 60)
    
    # ===== 1. 显示评估指标 =====
    try:
        with open('tdnn_eval_results_no_weather.json', 'r') as f:
            results = json.load(f)
        print("\n📊 模型评估指标（来自验证集）:")
        print(f"   R² : {results.get('r2', 0):.4f}")
        print(f"   RMSE: {results.get('rmse', 0):.2f} 元/MWh")
        print(f"   MAE : {results.get('mae', 0):.2f} 元/MWh")
        print(f"   方向准确率: {results.get('direction_accuracy', 0):.2%}")
    except FileNotFoundError:
        print("\n⚠️ 未找到评估结果文件")
    
    # ===== 2. 加载模型 =====
    print("\n📥 加载模型...")
    model = TDNN(config)
    model_path = 'tdnn_best_model_no_weather.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行训练脚本")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    print(f"✅ 模型加载成功: {model_path}")
    
    # ===== 3. 获取最近数据 =====
    print("\n📥 获取最近数据...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    df = dataset(
        province=config.province,
        start_date=start_date,
        end_date=end_date
    )
    
    if len(df) == 0:
        print("❌ 没有获取到数据")
        return None
    
    print(f"   数据范围: {start_date} ~ {end_date}")
    print(f"   数据行数: {len(df)}")
    
    # ===== 4. 准备输入 =====
    dataset_obj = PriceSpreadDataset(df, config)
    
    feature_data = df[config.feature_cols].values.astype(np.float32)
    feature_data = np.nan_to_num(feature_data, nan=0.0)
    feature_data = (feature_data - dataset_obj.feature_min) / dataset_obj.feature_range
    
    if len(feature_data) < config.input_len:
        print(f"⚠️ 数据不足！需要 {config.input_len} 个点，只有 {len(feature_data)} 个")
        return None
    
    X = feature_data[-config.input_len:]
    X = torch.FloatTensor(X).unsqueeze(0).to(config.device)
    
    # ===== 5. 预测 =====
    print("\n🔮 预测未来1天96点价差...")
    with torch.no_grad():
        pred_norm = model(X).cpu().numpy()[0]
    
    pred = pred_norm * dataset_obj.target_range + dataset_obj.target_min
    
    # ===== 6. 输出结果 =====
    pred_date = datetime.now() + timedelta(days=1)
    pred_date_str = pred_date.strftime('%Y-%m-%d')
    
    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)]
    
    result_df = pd.DataFrame({
        'timestamp': time_labels,
        'price_spread': pred
    })
    
    print(f"\n📊 预测日期: {pred_date_str}")
    print(f"\n价差统计:")
    print(f"   最小值: {pred.min():.2f} 元/MWh")
    print(f"   最大值: {pred.max():.2f} 元/MWh")
    print(f"   平均值: {pred.mean():.2f} 元/MWh")
    print(f"   标准差: {pred.std():.2f} 元/MWh")
    
    # ===== 7. 保存CSV =====
    output_csv = f'tdnn_prediction_{pred_date_str}.csv'
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV 已保存: {output_csv}")
    
    # ===== 8. 绘图 =====
    plt.figure(figsize=(14, 5))
    plt.plot(range(96), pred, linewidth=2, color='blue', label='预测价差')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='零线')
    plt.fill_between(range(96), pred, 0, alpha=0.1, color='blue')
    plt.title(f'TDNN 未来1天价差预测 ({pred_date_str})', fontsize=14)
    plt.xlabel('时间点 (15分钟间隔)', fontsize=12)
    plt.ylabel('price_spread (元/MWh)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_png = f'tdnn_prediction_{pred_date_str}.png'
    plt.savefig(output_png, dpi=300)
    print(f"📊 曲线图已保存: {output_png}")
    plt.show()
    
    # ===== 9. 打印前10个点 =====
    print("\n" + "=" * 60)
    print("前10个时间点预测值")
    print("=" * 60)
    print(result_df.head(10).to_string(index=False))
    
    return pred, result_df


if __name__ == '__main__':
    predict_next_day()