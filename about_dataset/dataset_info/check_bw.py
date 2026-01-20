import json
import os
import glob
from collections import defaultdict

# ==========================================
# 1. 配置参数
# ==========================================

DATASET_DIR = '/data/hwh_data_folder/dataset/train'  # 数据集标签路径
REPORT_FILE = 'full_bandwidth_report.txt'            # 输出报告文件名

# 定义关注的类别及其“标准”带宽 (MHz)
# TARGET_CONFIG = {
#     3:  {'name': 'WiFi_40_QPSK',    'expected_bw': 40.0},
#     9:  {'name': 'Lora_0.25',       'expected_bw': 0.25},
#     10: {'name': 'Custom_10_QPSK',  'expected_bw': 10.0},
#     11: {'name': 'Custom_10_16QAM', 'expected_bw': 10.0},
#     12: {'name': 'Custom_0.2_AM',   'expected_bw': 0.2},
#     13: {'name': 'Custom_0.2_FM',   'expected_bw': 0.2}
# }
TARGET_CONFIG = {
    0:  {'name': 'WiFi_20_QPSK',    'expected_bw': 20.0},
    1:  {'name': 'WiFi_20_16QAM',    'expected_bw': 20.0},
    2:  {'name': 'WiFi_20_64QAM',    'expected_bw': 20.0},
    3:  {'name': 'WiFi_40_QPSK',    'expected_bw': 40.0},
    4:  {'name': 'WiFi_40_16QAM',    'expected_bw': 40.0},
    5:  {'name': 'WiFi_40_64QAM',    'expected_bw': 40.0},
    6:  {'name': 'BLE_1_LE',    'expected_bw': 1},
    7:  {'name': 'BLE_2_LE',    'expected_bw': 2},
    8:  {'name': 'ZigBee_2_OQPSK',    'expected_bw': 2},
    9:  {'name': 'Lora_0.25',       'expected_bw': 0.25},
    10: {'name': 'Custom_10_QPSK',  'expected_bw': 10.0},
    11: {'name': 'Custom_10_16QAM', 'expected_bw': 10.0},
    12: {'name': 'Custom_0.2_AM',   'expected_bw': 0.2},
    13: {'name': 'Custom_0.2_FM',   'expected_bw': 0.2}
}

# 浮点数判定容差
TOLERANCE = 0.001 

# ==========================================
# 2. 核心处理逻辑
# ==========================================

def analyze_bandwidths():
    json_files = glob.glob(os.path.join(DATASET_DIR, '*.json'))
    # 按文件名数字排序
    json_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    print(f"开始扫描 {len(json_files)} 个标签文件...")

    # 数据结构: results[class_id][bandwidth] = [file_id1, file_id2, ...]
    results = defaultdict(lambda: defaultdict(list))
    
    for file_path in json_files:
        file_idx = int(os.path.basename(file_path).split('.')[0])
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        signals = data.get('signals', [])
        
        for sig in signals:
            cls = sig['class']
            
            # 只处理关注的类别
            if cls in TARGET_CONFIG:
                # 计算带宽并保留4位小数
                bw = sig['end_frequency'] - sig['start_frequency']
                bw_key = round(bw, 4)
                
                # 记录文件ID (同一个文件如果出现多次相同带宽的同类信号，只记一次)
                if file_idx not in results[cls][bw_key]:
                    results[cls][bw_key].append(file_idx)

    # ==========================================
    # 3. 生成详细报告
    # ==========================================
    
    print("扫描完成，正在生成对比报告...")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("Dataset Bandwidth Full Analysis Report\n")
        f.write("======================================\n")
        f.write(f"Data Source: {DATASET_DIR}\n\n")

        # 按 Class ID 顺序输出
        for cls in sorted(results.keys()):
            info = TARGET_CONFIG[cls]
            expected_val = info['expected_bw']
            
            f.write(f"### Class {cls} ({info['name']})\n")
            f.write(f"Target Standard Bandwidth: {expected_val} MHz\n")
            f.write("-" * 50 + "\n")
            
            # 获取该类下统计到的所有带宽种类
            found_bws = results[cls]
            
            # 按带宽数值从小到大排序显示
            for bw_val in sorted(found_bws.keys()):
                file_list = found_bws[bw_val]
                count = len(file_list)
                
                # 判断是标准还是异常
                is_standard = abs(bw_val - expected_val) <= TOLERANCE
                
                if is_standard:
                    tag = "[STANDARD] (标准)"
                    # 为了排版美观，标准通常可能想放前面，或者用星号标记
                else:
                    tag = f"[ANOMALY] (异常! 差值: {bw_val - expected_val:.4f})"
                
                f.write(f"  ● Bandwidth: {bw_val} MHz  {tag}\n")
                f.write(f"    Count: {count} files\n")
                f.write(f"    IDs: {file_list}\n") # 打印所有ID
                f.write("\n")
            
            f.write("\n" + "="*50 + "\n\n")

    print(f"报告已保存至: {os.path.abspath(REPORT_FILE)}")

if __name__ == "__main__":
    analyze_bandwidths()