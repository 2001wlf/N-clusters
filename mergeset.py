import pickle
import os
import re
import numpy as np
from tqdm import tqdm

def merge_dict_structure_files(input_folder, output_path):
    # 1. 筛选文件（保持之前的正则逻辑）
    pattern = re.compile(r'^\d+_[a-zA-Z]+_part_\d+\.pkl$')
    
    if not os.path.exists(input_folder):
        print(f"文件夹不存在: {input_folder}")
        return

    # 获取并排序文件
    files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if pattern.match(f)])
    
    if not files:
        print("未找到符合格式的文件。")
        return

    print(f"准备合并 {len(files)} 个文件...")

    # 2. 初始化缓存容器
    # 我们需要一个字典，key对应您数据的key，value是一个列表，用来暂存所有文件的数组
    # 例如: buffer = { "node_feat": [arr1, arr2...], "edge_feat": [arr1, arr2...] }
    buffer = {} 
    keys = []

    # 3. 读取第一个文件以初始化 Key
    print("正在读取文件并收集数据...")
    
    # 我们先读取第一个文件来确定有哪些 key
    with open(files[0], 'rb') as f:
        first_data = pickle.load(f)
        keys = list(first_data.keys())
        # 初始化 buffer
        for k in keys:
            buffer[k] = []

    # 4. 循环读取所有文件
    for file_path in tqdm(files, desc="Loading"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                # 将该文件的每个特征数组，添加到对应的 buffer 列表中
                for k in keys:
                    buffer[k].append(data[k])
                    
        except Exception as e:
            print(f"跳过损坏文件 {file_path}: {e}")

    # 5. 执行拼接 (Concatenate)
    print("正在进行最终拼接 (Numpy Concatenate)...")
    final_dict = {}
    
    for k in tqdm(keys, desc="Concat"):
        # axis=0 表示沿着第一个维度（样本数 n_samples）进行拼接
        # 这会将多个 (100, 1000, 2) 的数组拼成一个 (50000, 1000, 2) 的大数组
        final_dict[k] = np.concatenate(buffer[k], axis=0)

    # 6. 验证并保存
    n_total = final_dict[keys[0]].shape[0] # 获取总样本数
    print(f"拼接完成！总样本数: {n_total}")
    print(f"正在保存到 {output_path} (可能需要几分钟)...")
    
    # 使用 protocol=4 或 5 以支持大文件（>4GB）
    with open(output_path, 'wb') as f:
        pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("保存成功！")

# ================= 运行配置 =================
if __name__ == "__main__":
    INPUT = r'C:/Users/10998/Desktop/N-clusters/density/mydata/val'              # 您的输入文件夹
    OUTPUT = r'C:/Users/10998/Desktop/N-clusters/density/mydata/val/val_merged.pkl'  # 输出文件
    
    merge_dict_structure_files(INPUT, OUTPUT)