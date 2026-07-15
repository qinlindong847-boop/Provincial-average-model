"""
数据加载与处理模块
用于加载广东电力市场数据，支持 TDNN 训练
"""

import pandas as pd
import numpy as np
import json
import re
from chinese_calendar import is_workday

# 从 tlzn_data 导入你已有的接口
from tlzn_data import (
    query_spot_renewable_output_data,
    query_historical_forecast_weather_data
)
from tlzn_data.utils import (
    _get_mysql_conn_pro,
    _get_mysql_conn, 
    _close_mysql_conn, 
    _transpose_electricity_data_df, 
    _fill_complete_timestamps
)
from tlzn_data.apis.util_api import prov2pinyin


# ===================== 1. 负荷数据查询 =====================
def query_load_data_fixed(province: str, dtype: str, start_date: str, end_date: str = None):
    """
    直接查询负荷数据
    """
    province = prov2pinyin(province)
    end_date = end_date if end_date else start_date
    
    if province != 'guangdong':
        raise ValueError(f"province {province} not supported yet")
    
    all_cols = ['统调负荷', '省内A类电源', '省内B类电源', '地方电源出力', '西电东送电力', '粤港联络线']
    data_type = 0 if dtype == '日前' else 1
    table_name = 'electricity_load_info'
    
    try:
        conn, cursor = _get_mysql_conn_pro()
        
        placeholders = ','.join(['%s'] * len(all_cols))
        sql = f"""
            SELECT record_date, load_type, load_data
            FROM {table_name}
            WHERE data_type = %s
              AND load_type IN ({placeholders})
              AND record_date BETWEEN %s AND %s
        """
        params = [data_type] + all_cols + [start_date, end_date]
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        if not results:
            return pd.DataFrame(columns=['record_date', 'timestamp'] + all_cols)
        
        data_df = pd.DataFrame(results)
        data_df.columns = ['record_date', 'load_type', 'load_data']
        data_df['record_date'] = data_df['record_date'].astype(str)
        
        data_df = _transpose_electricity_data_df(
            data_df, 
            date_col='record_date', 
            key_col='load_type', 
            val_col='load_data',
            reserved_key_col=all_cols
        )
        
        data_df = _fill_complete_timestamps(
            data_df, start_date, end_date, 
            date_col='record_date', 
            data_cols=all_cols, 
            freq='15min'
        )
        
        for col in all_cols:
            data_df[col] = data_df[col].ffill().bfill()
        
        return data_df[['record_date', 'timestamp'] + all_cols]
        
    except Exception as e:
        raise
    finally:
        if 'conn' in locals() and 'cursor' in locals():
            _close_mysql_conn(conn, cursor)


# ===================== 2. 全省平均电价查询 =====================
def query_province_avg_price_data(province: str, dtype: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    查询全省平均电价（所有场站的平均值，uuid=0）
    """
    province = prov2pinyin(province)
    end_date = end_date if end_date else start_date
    
    if province != 'guangdong':
        raise ValueError(f"province {province} does not have province average price table")
    
    if dtype == '日前':
        table_name = 'gd_electricity_price_allclassity_ahead'
    elif dtype == '实时':
        table_name = 'gd_electricity_price_allclassity_real'
    else:
        raise ValueError(f"dtype must be '日前' or '实时', got {dtype}")
    
    uuid = '0'
    
    try:
        conn, cursor = _get_mysql_conn()
        
        sql = f"""
            SELECT uuid, record_date, price_data
            FROM {table_name}
            WHERE uuid = %s
              AND record_date BETWEEN %s AND %s
        """
        cursor.execute(sql, (uuid, start_date, end_date))
        results = cursor.fetchall()
        
        data_df = pd.DataFrame(results)
        if data_df.empty:
            return pd.DataFrame(columns=['record_date', 'timestamp', 'price_data', 'uuid'])
        
        data_df['record_date'] = data_df['record_date'].astype(str)
        data_df = _transpose_electricity_data_df(
            data_df, 
            date_col='record_date', 
            key_col='uuid', 
            val_col='price_data'
        )
        data_df = _fill_complete_timestamps(
            data_df, start_date, end_date, 
            date_col='record_date', 
            data_cols=[uuid], 
            freq='15min'
        )
        data_df['uuid'] = uuid
        data_df = data_df.rename(columns={uuid: 'price_data'})
        
        return data_df[['record_date', 'timestamp', 'price_data', 'uuid']]
        
    except Exception as e:
        raise
    finally:
        if 'conn' in locals() and 'cursor' in locals():
            _close_mysql_conn(conn, cursor)


# ===================== 3. 全省平均天气查询（多个气象站平均） =====================
def query_province_avg_weather(province: str, start_date: str, end_date: str, data_cols: list) -> pd.DataFrame:
    """
    查询全省平均天气：用多个气象站的数据取平均值
    
    广东省主要气象站（覆盖不同区域）：
    - 广州：中部
    - 韶关：北部
    - 汕头：东部
    - 湛江：西部
    - 深圳：南部
    """
    province = prov2pinyin(province)
    
    # 广东省主要气象站（覆盖全省不同区域）
    stations = [
        '广州气象站',   # 中部
        '韶关气象站',   # 北部
        '汕头气象站',   # 东部
        '湛江气象站',   # 西部
        '深圳气象站',   # 南部
    ]
    
    all_dfs = []
    valid_stations = []
    
    for station in stations:
        try:
            df = query_historical_forecast_weather_data(
                province=province,
                station_name=station,
                start_date=start_date,
                end_date=end_date,
                data_cols=data_cols,
                parse_data=True,
                strict=False
            )
            if not df.empty:
                all_dfs.append(df)
                valid_stations.append(station)
                print(f"   ✅ {station} 数据加载成功")
        except Exception as e:
            print(f"   ⚠️ {station} 数据加载失败: {e}")
            continue
    
    if not all_dfs:
        raise ValueError("没有获取到任何气象站数据")
    
    print(f"   📊 使用 {len(valid_stations)} 个气象站: {valid_stations}")
    
    # 合并所有气象站数据，按时间取平均
    base_df = all_dfs[0][['timestamp'] + data_cols].copy()
    base_df = base_df.rename(columns={col: f'{col}_0' for col in data_cols})
    
    for i, df in enumerate(all_dfs[1:], 1):
        df_subset = df[['timestamp'] + data_cols].copy()
        df_subset = df_subset.rename(columns={col: f'{col}_{i}' for col in data_cols})
        base_df = base_df.merge(df_subset, on='timestamp', how='outer')
    
    # 取平均值
    for col in data_cols:
        col_versions = [c for c in base_df.columns if c.startswith(f'{col}_')]
        if col_versions:
            base_df[col] = base_df[col_versions].mean(axis=1)
    
    # 只保留 timestamp 和平均后的列
    result_df = base_df[['timestamp'] + data_cols].copy()
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)
    
    # 填充缺失值
    for col in data_cols:
        result_df[col] = result_df[col].ffill().bfill().fillna(0)
    
    return result_df


# ===================== 4. 阻塞数据查询（带量化） =====================
def query_block_data_fixed(province: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    province = prov2pinyin(province)
    end_date = end_date if end_date else start_date
    
    if province != 'guangdong':
        return pd.DataFrame(columns=['timestamp', 'block_weight'])
    
    table_name = 'electricity_block_info'
    
    try:
        conn, cursor = _get_mysql_conn_pro()
        
        sql = f"""
            SELECT record_date, block_info, overline_section
            FROM {table_name}
            WHERE data_type IN (0, 1)
              AND record_date BETWEEN %s AND %s
        """
        cursor.execute(sql, (start_date, end_date))
        results = cursor.fetchall()
        
        if not results:
            timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
            return pd.DataFrame({'timestamp': timestamps, 'block_weight': 0})
        
        data_df = pd.DataFrame(results)
        data_df.columns = ['record_date', 'block_info', 'overline_section']
        
        def calc_block_weight(row):
            info = row['block_info']
            if pd.isna(info) or info is None:
                return 0
            text = str(info)
            weight = 0
            
            if '500kV' in text:
                weight += 2
            elif '220kV' in text:
                weight += 1
            
            if '主变跳闸' in text:
                weight += 2
            elif '一回跳闸' in text or '线路跳闸' in text:
                weight += 1
            elif '跳闸' in text:
                weight += 1
            
            if '故障' in text and '跳闸' not in text:
                weight += 1
            
            numbers = re.findall(r'(\d+)', text)
            if numbers:
                max_num = max(map(int, numbers))
                if max_num > 2000:
                    weight += 1
                elif max_num > 1000:
                    weight += 0.5
            
            return min(weight, 5)
        
        data_df['block_weight'] = data_df.apply(calc_block_weight, axis=1)
        
        # 调试：打印量化结果
        print(f"[DEBUG] 原始记录数: {len(data_df)}")
        print(f"[DEBUG] 量化后非零记录数: {(data_df['block_weight'] > 0).sum()}")
        if (data_df['block_weight'] > 0).sum() > 0:
            print("[DEBUG] 示例:")
            print(data_df[data_df['block_weight'] > 0][['record_date', 'block_info', 'block_weight']].head(3))
        
        daily_weight = data_df.groupby('record_date')['block_weight'].max().reset_index()
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
        result_df = pd.DataFrame({'timestamp': timestamps})
        result_df['record_date'] = result_df['timestamp'].dt.strftime('%Y-%m-%d')
        
        result_df = result_df.merge(daily_weight, on='record_date', how='left')
        result_df['block_weight'] = result_df['block_weight'].fillna(0)
        
        return result_df[['timestamp', 'block_weight']]
        
    except Exception as e:
        print(f'错误: {e}')
        raise
    finally:
        if 'conn' in locals() and 'cursor' in locals():
            _close_mysql_conn(conn, cursor)

def query_maintenance_data_fixed(province: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    province = prov2pinyin(province)
    end_date = end_date if end_date else start_date
    
    if province != 'guangdong':
        return pd.DataFrame(columns=['timestamp', 'maintenance_weight'])
    
    table_name = 'electricity_substation_maintenance_info'
    
    try:
        conn, cursor = _get_mysql_conn_pro()
        
        sql = f"""
            SELECT record_date, voltage_level
            FROM {table_name}
            WHERE data_type IN (0, 1)
              AND record_date BETWEEN %s AND %s
        """
        cursor.execute(sql, (start_date, end_date))
        results = cursor.fetchall()
        
        if not results:
            timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
            return pd.DataFrame({'timestamp': timestamps, 'maintenance_weight': 0})
        
        data_df = pd.DataFrame(results)
        data_df.columns = ['record_date', 'voltage_level']
        
        # 量化：500kV→4, 220kV→2, 其他→1
        data_df['kv_level'] = data_df['voltage_level'].map({500: 4, 220: 2}).fillna(1)
        
        # 按天聚合
        daily_weight = data_df.groupby('record_date')['kv_level'].sum().reset_index()
        daily_weight.columns = ['record_date', 'maintenance_weight']
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
        result_df = pd.DataFrame({'timestamp': timestamps})
        result_df['record_date'] = result_df['timestamp'].dt.strftime('%Y-%m-%d')
        
        result_df = result_df.merge(daily_weight, on='record_date', how='left')
        result_df['maintenance_weight'] = result_df['maintenance_weight'].fillna(0)
        
        # 调试打印
        print(f"[DEBUG maintenance] 原始记录数: {len(data_df)}")
        print(f"[DEBUG maintenance] 量化后非零记录数: {(data_df['kv_level'] > 0).sum()}")
        if (data_df['kv_level'] > 0).sum() > 0:
            print("[DEBUG maintenance] 示例:")
            print(data_df[['record_date', 'voltage_level', 'kv_level']].head(5))
        
        return result_df[['timestamp', 'maintenance_weight']]
        
    except Exception as e:
        print(f'错误: {e}')
        raise
    finally:
        if 'conn' in locals() and 'cursor' in locals():
            _close_mysql_conn(conn, cursor)


# ===================== 6. 主数据集（96点/天细粒度） =====================
def dataset(province='广东', start_date='2024-01-01', end_date='2026-06-30'):
    """
    加载并处理所有数据，返回96点/天的细粒度数据集
    """
    # ===== 负荷数据处理 =====
    all_cols = ['统调负荷', '省内A类电源', '省内B类电源', '地方电源出力', '西电东送电力', '粤港联络线']
    
    df_ahead_load = query_load_data_fixed(province, '日前', start_date, end_date)
    df_real_load = query_load_data_fixed(province, '实时', start_date, end_date)

    if 'record_date' in df_ahead_load.columns:
        df_ahead_load = df_ahead_load.drop(columns='record_date')
    if 'record_date' in df_real_load.columns:
        df_real_load = df_real_load.drop(columns='record_date')

    df_ahead_load['省内B类电源'] += df_ahead_load['西电东送电力'].fillna(
        df_ahead_load['统调负荷'] - df_ahead_load[['地方电源出力', '省内A类电源', '省内B类电源', '粤港联络线']].sum(axis=1)
    )
    df_ahead_load = df_ahead_load.drop(columns='西电东送电力')

    df_ahead_load = df_ahead_load.rename(columns=lambda c: f'ahead_{c}' if c != 'timestamp' else c)
    df_real_load = df_real_load.rename(columns=lambda c: f'real_{c}' if c != 'timestamp' else c)
    
    df_final_load = pd.merge(df_ahead_load, df_real_load, on='timestamp', how='outer')
    df_final_load = df_final_load.sort_values('timestamp').reset_index(drop=True)
    df_final_load = df_final_load.ffill().bfill()

    # ===== 价格数据处理（全省平均，uuid=0） =====
    df_ahead_price = query_province_avg_price_data(province, '日前', start_date, end_date)
    df_real_price = query_province_avg_price_data(province, '实时', start_date, end_date)
    
    df_ahead_price = df_ahead_price.drop(columns='uuid').rename(columns={'price_data': 'ahead_price_data'})
    df_real_price = df_real_price.drop(columns='uuid').rename(columns={'price_data': 'real_price_data'})
    
    if 'record_date' in df_ahead_price.columns:
        df_ahead_price = df_ahead_price.drop(columns='record_date')
    if 'record_date' in df_real_price.columns:
        df_real_price = df_real_price.drop(columns='record_date')
    
    df_merge_price = pd.merge(df_ahead_price, df_real_price, on='timestamp', how='outer')
    df_merge_price = df_merge_price.sort_values('timestamp').reset_index(drop=True)

    # ===== 阻塞数据处理（带量化） =====
    df_block = query_block_data_fixed(province, start_date, end_date)

    # ===== 检修数据处理 =====
    df_maintenance = query_maintenance_data_fixed(province, start_date, end_date)

    # ===== 天气数据处理（可选，暂不使用） =====
    # 如果需要天气，取消注释以下代码
    # weather_cols = ['temperature_2m', 'precipitation', 'wind_speed_100m', 
    #                 'sunshine_duration', 'shortwave_radiation']
    # df_weather = query_province_avg_weather(province, start_date, end_date, weather_cols)

    # ===== 合并所有数据 =====
    dfs = {
        'df_final_load': df_final_load,
        'df_merge_price': df_merge_price,
        'df_block': df_block,
        'df_maintenance': df_maintenance,
        # 'df_weather': df_weather  # 需要天气时取消注释
    }

    for name, df in dfs.items():
        if 'timestamp' not in df.columns:
            raise ValueError(f"{name} missing 'timestamp' column: {df.columns.tolist()}")
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df_final = (df_final_load
                .merge(df_merge_price, on='timestamp', how='outer')
                .merge(df_block, on='timestamp', how='outer')
                .merge(df_maintenance, on='timestamp', how='outer')
                # .merge(df_weather, on='timestamp', how='outer')  # 需要天气时取消注释
                .sort_values('timestamp')
                .reset_index(drop=True)
                )

    df_final['price_spread'] = df_final['ahead_price_data'] - df_final['real_price_data']
    df_final['is_workday'] = df_final['timestamp'].dt.date.apply(is_workday).astype(int)

    return df_final


# ===================== 7. 测试 =====================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 dataset() 细粒度数据（含阻塞量化）")
    print("=" * 60)
    
    df_test = dataset(
        province='广东',
        start_date='2024-01-01',
        end_date='2024-01-07'
    )
    
    print(f"数据行数: {len(df_test)} (应为 672行)")
    print(f"列数: {len(df_test.columns)}")
    print("\n前5行:")
    print(df_test[['timestamp', 'ahead_price_data', 'real_price_data', 'price_spread', 'block_weight']].head())
    
    # 检查阻塞数据
    block_nonzero = df_test[df_test['block_weight'] > 0]
    print(f"\n阻塞事件数量: {len(block_nonzero)}")
    if len(block_nonzero) > 0:
        print(block_nonzero[['timestamp', 'block_weight']].head())
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)