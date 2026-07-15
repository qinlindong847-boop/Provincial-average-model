"""
XGBoost MultiOutput 训练脚本
输出：未来 1 天的 96 个价差点
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
import matplotlib.pyplot as plt
import joblib
import os

warnings.filterwarnings('ignore')

from dataset import dataset_for_xgb_96


def train_xgb_multioutput():
    print("=" * 60)
    print("XGBoost MultiOutput 训练 (96点价差预测)")
    print("=" * 60)
    
    # ===== 加载数据 =====
    print("\n📥 加载数据...")
    
    X, y_96, feature_cols, dates = dataset_for_xgb_96(
        province='广东',
        start_date='2024-01-01',
        end_date='2025-12-31',
        window=7
    )
    
    print(f"样本数: {len(X)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"目标维度: {y_96.shape[1]} 个点/天")
    print(f"日期范围: {dates[0]} ~ {dates[-1]}")
    
    # ===== 按时间划分 =====
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_96[:split_idx], y_96[split_idx:]
    dates_train, dates_val = dates[:split_idx], dates[split_idx:]
    
    print(f"\n训练集: {len(X_train)} 样本 ({dates_train[0]} ~ {dates_train[-1]})")
    print(f"验证集: {len(X_val)} 样本 ({dates_val[0]} ~ {dates_val[-1]})")
    
    # ===== 训练 MultiOutput XGBoost =====
    print("\n" + "=" * 60)
    print("🚀 训练 XGBoost MultiOutput...")
    print("=" * 60)
    print("注意: MultiOutput 会为每个时间点训练一个模型 (共96个)")
    print("预计耗时: 5-15 分钟 (取决于 CPU 核心数)\n")
    
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42
    )
    
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("✅ 训练完成！")
    
    # ===== 保存模型 =====
    print("\n💾 保存模型...")
    joblib.dump(model, 'xgb_multioutput_model.pkl')
    with open('xgb_multioutput_feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    print("✅ 模型已保存: xgb_multioutput_model.pkl")
    
    # ===== 评估 =====
    print("\n" + "=" * 60)
    print("📊 模型评估")
    print("=" * 60)
    
    y_pred = model.predict(X_val)
    
    # 整体指标
    rmse_all = np.sqrt(mean_squared_error(y_val.flatten(), y_pred.flatten()))
    mae_all = mean_absolute_error(y_val.flatten(), y_pred.flatten())
    r2_all = r2_score(y_val.flatten(), y_pred.flatten())
    
    print(f"整体 RMSE: {rmse_all:.4f} 元/MWh")
    print(f"整体 MAE : {mae_all:.4f} 元/MWh")
    print(f"整体 R²  : {r2_all:.4f}")
    
    # 逐点 RMSE
    point_rmse = []
    for i in range(96):
        rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i]))
        point_rmse.append(rmse)
    
    print(f"\n各时间点 RMSE 统计:")
    print(f"  最小: {min(point_rmse):.4f}")
    print(f"  最大: {max(point_rmse):.4f}")
    print(f"  平均: {np.mean(point_rmse):.4f}")
    
    # 逐点 RMSE 绘图
    plt.figure(figsize=(14, 5))
    plt.plot(range(96), point_rmse, linewidth=2, color='blue')
    plt.axhline(y=np.mean(point_rmse), color='red', linestyle='--', label=f'平均: {np.mean(point_rmse):.2f}')
    plt.xlabel('时间点 (15分钟间隔)')
    plt.ylabel('RMSE (元/MWh)')
    plt.title('各时间点 RMSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgb_multioutput_point_rmse.png', dpi=300)
    print("📊 逐点 RMSE 图已保存: xgb_multioutput_point_rmse.png")
    
    # ===== 特征重要性 =====
    importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': model.estimators_[0].feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n" + "=" * 60)
    print("📊 特征重要性 (Top 10)")
    print("=" * 60)
    print(importance.head(10))
    
    # ===== 绘图: 某一天的预测曲线 =====
    sample_idx = -1
    plt.figure(figsize=(14, 6))
    plt.plot(y_val[sample_idx], label='真实价差', linewidth=2, color='blue')
    plt.plot(y_pred[sample_idx], label='预测价差', linewidth=2, alpha=0.8, color='orange')
    plt.title(f'96点价差预测对比 (日期: {dates_val[sample_idx]})')
    plt.xlabel('时间点 (15分钟间隔)')
    plt.ylabel('price_spread (元/MWh)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgb_multioutput_96points.png', dpi=300)
    print("📊 96点预测图已保存: xgb_multioutput_96points.png")
    plt.show()
    
    # ===== 输出示例 =====
    print("\n" + "=" * 60)
    print("📋 预测输出示例（验证集最后一天）")
    print("=" * 60)
    print(f"日期: {dates_val[sample_idx]}")
    print(f"前10个点预测值: {y_pred[sample_idx][:10]}")
    print(f"前10个点真实值: {y_val[sample_idx][:10]}")
    
    # ===== 保存评估结果 =====
    eval_results = {
        'rmse_all': float(rmse_all),
        'mae_all': float(mae_all),
        'r2_all': float(r2_all),
        'point_rmse_mean': float(np.mean(point_rmse)),
        'point_rmse_min': float(min(point_rmse)),
        'point_rmse_max': float(max(point_rmse)),
        'feature_importance': importance.head(10).to_dict()
    }
    
    with open('xgb_multioutput_eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n✅ 评估结果已保存: xgb_multioutput_eval_results.json")
    
    return model


if __name__ == '__main__':
    train_xgb_multioutput()