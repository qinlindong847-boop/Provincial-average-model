import re

with open('dataset.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到函数开始和结束
start_idx = None
end_idx = None
depth = 0

for i, line in enumerate(lines):
    if line.strip().startswith('def query_block_data_fixed('):
        start_idx = i
        depth = 0
    elif start_idx is not None:
        # 检查缩进变化来找到函数结束
        if line.strip() and not line.startswith((' ', '\t')) and line.strip() and not line.strip().startswith('#'):
            if depth == 0:
                end_idx = i
                break
        # 计数花括号
        depth += line.count('{') - line.count('}')

# 如果没找到结束，到文件末尾
if end_idx is None:
    end_idx = len(lines)

print(f'找到函数: 第 {start_idx+1} 行到第 {end_idx+1} 行')

# 新函数内容
new_func = '''def query_block_data_fixed(province: str, start_date: str, end_date: str = None) -> pd.DataFrame:
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

'''

if start_idx is not None:
    # 替换函数
    new_lines = lines[:start_idx] + new_func.splitlines(keepends=True) + lines[end_idx:]
    with open('dataset.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print('✅ 函数已替换')
else:
    print('❌ 未找到函数')
