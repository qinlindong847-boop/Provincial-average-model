import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from dataset import dataset
from tdnn_train import Config, TDNN, PriceSpreadDataset


# ===================== 辅助函数 =====================
def calculate_direction_accuracy(y_true, y_pred):
    """
    计算方向准确率：预测的涨跌方向与实际是否一致
    """
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
    """
    计算 MAPE（平均绝对百分比误差）
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) == 0:
        return float('inf')
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def predict_next_day():
    """
    预测未来1天的96点价差
    """
    config = Config()
    print("=" * 60)
    print("TDNN 电价价差预测")
    print("=" * 60)
    
    # ===== 1. 加载模型和评估结果 =====
    print("\n📥 加载模型...")
    model = TDNN(config)
    
    try:
        model.load_state_dict(torch.load('tdnn_best_model.pt', map_location=config.device))
        model.to(config.device)
        model.eval()
        print("✅ 模型加载成功: tdnn_best_model.pt")
    except FileNotFoundError:
        print("❌ 模型文件不存在！请先运行 tdnn_train.py 训练模型")
        return None
    
    # 加载评估结果
    try:
        with open('tdnn_eval_results.json', 'r') as f:
            eval_results = json.load(f)
        print("\n📊 模型评估指标（来自验证集）:")
        print(f"   R² : {eval_results.get('r2', 'N/A'):.4f}")
        print(f"   RMSE: {eval_results.get('rmse', 'N/A'):.2f} 元/MWh")
        print(f"   MAE : {eval_results.get('mae', 'N/A'):.2f} 元/MWh")
        dir_acc = eval_results.get('direction_accuracy', None)
        if dir_acc:
            print(f"   方向准确率: {dir_acc:.2%}")
        mape = eval_results.get('mape', None)
        if mape:
            print(f"   MAPE : {mape:.2f}%")
    except FileNotFoundError:
        print("\n⚠️ 未找到评估结果文件，请先运行 tdnn_train.py")
    
    # ===== 2. 获取最近的数据 =====
    print("\n📥 获取最近数据...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    print(f"   数据范围: {start_date} ~ {end_date}")
    
    df = dataset(
        province=config.province,
        start_date=start_date,
        end_date=end_date,
        classify_id=config.classify_id,
        station_name=config.station_name
    )
    
    if len(df) == 0:
        print("❌ 没有获取到数据")
        return None
    
    print(f"   获取数据行数: {len(df)}")
    
    # ===== 3. 创建数据集对象（获取归一化参数） =====
    dataset_obj = PriceSpreadDataset(df, config)
    
    # ===== 4. 提取最近7天的特征 =====
    feature_data = df[config.feature_cols].values.astype(np.float32)
    feature_data = np.nan_to_num(feature_data, nan=0.0)
    
    feature_data = (feature_data - dataset_obj.feature_min) / dataset_obj.feature_range
    
    if len(feature_data) < config.input_len:
        print(f"⚠️ 数据不足！需要 {config.input_len} 个点，只有 {len(feature_data)} 个")
        return None
    
    X = feature_data[-config.input_len:]  # (672, 9)
    X = torch.FloatTensor(X).unsqueeze(0).to(config.device)  # (1, 672, 9)
    
    # ===== 5. 预测 =====
    print("\n🔮 预测未来1天96点价差...")
    with torch.no_grad():
        pred_normalized = model(X).cpu().numpy()[0]  # (96,)
    
    pred = pred_normalized * dataset_obj.target_range + dataset_obj.target_min
    
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
    print(f"  最小值: {pred.min():.2f} 元/MWh")
    print(f"  最大值: {pred.max():.2f} 元/MWh")
    print(f"  平均值: {pred.mean():.2f} 元/MWh")
    print(f"  标准差: {pred.std():.2f} 元/MWh")
    
    # ===== 7. 保存 CSV =====
    output_csv = f'tdnn_prediction_{pred_date_str}.csv'
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV 已保存: {output_csv}")
    
    # ===== 8. 绘制曲线图 =====
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
    pred, result_df = predict_next_day()