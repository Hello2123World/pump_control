import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class DataNormalizer:
    def __init__(self):
        """
        初始化 DataNormalizer 类
        """
        pass

    def normalize_data(self, file_path, index, pressure):
        """
        读取文件并对数据进行归一化处理
        :param file_path: 输入数据文件路径
        :param index: 当前文件的索引
        :param pressure: 当前压力值（文件夹名称）
        """
        # 读取文件
        data = pd.read_csv(file_path)
        print(f"Processing file: {file_path}")
        
        # 如果数据为空，跳过
        if data.empty:
            print(f"Warning: {file_path} is empty. Skipping this file.")
            return

        # 定义需要归一化的列
        action_columns = ['pump_1_frequency', 'pump_2_frequency', 'pump_3_frequency', 'pump_4_frequency']
        state_columns = ['pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure', 'sum_pressure', 'target_pressure', 'flow']

        # 创建归一化器
        action_scaler = MinMaxScaler(feature_range=(-1, 1))  # 将频率数据归一化到[-1, 1]区间
        state_scaler = MinMaxScaler(feature_range=(-1, 1))  # 将状态数据归一化到[-1, 1]区间

        action_data = data[action_columns].values  # 转换为 numpy 数组
        state_data = data[state_columns].values  # 转换为 numpy 数组

        # 对action数据进行归一化
        normalized_action_data = action_scaler.fit_transform(action_data)

        # 对state数据进行归一化
        normalized_state_data = state_scaler.fit_transform(state_data)

        # 将归一化后的数据替换原始数据中的对应列
        new_data = data.copy()
        new_data[action_columns] = normalized_action_data
        new_data[state_columns] = normalized_state_data

        # 保存归一化后的数据到文件
        output_file_path = os.path.join(os.getcwd(), f'{pressure}/{index}_normalized.csv')
        new_data.to_csv(output_file_path, index=False)
        print(f"Normalized data saved to: {output_file_path}")

        # 保存归一化器
        scaler_dir = os.path.join(os.getcwd(), f"../scaler/{pressure}")
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)

        action_scaler_path = os.path.join(scaler_dir, f'{index}_action.pkl')
        state_scaler_path = os.path.join(scaler_dir, f'{index}_state.pkl')

        joblib.dump(action_scaler, action_scaler_path)
        joblib.dump(state_scaler, state_scaler_path)

        print(f"Scalers saved: {action_scaler_path}, {state_scaler_path}")

    def process_all_files(self, parent_folder):
        """
        遍历文件夹下的所有子文件夹（压力值文件夹），并执行归一化操作
        :param parent_folder: 主文件夹路径
        """
        for folder_name in os.listdir(parent_folder):
            folder_path = os.path.join(parent_folder, folder_name)
            
            if os.path.isdir(folder_path):
                pressure = folder_name  # 将文件夹名称作为压力值
                print(f"Processing data for pressure: {pressure}")

                original_folder_path = os.path.join(folder_path, 'original')
                # 检查 original 文件夹是否存在
                if os.path.exists(original_folder_path) and os.path.isdir(original_folder_path):
                    print(f"Found original folder in: {pressure}")
                    
                    # 遍历 "original" 文件夹中的所有 CSV 文件
                    for file_name in os.listdir(original_folder_path):
                        if file_name.endswith('.csv'):
                            file_path = os.path.join(original_folder_path, file_name)
                            index = file_name.split('.')[0]  # 从文件名提取索引
                            self.normalize_data(file_path, index, pressure)
                else:
                    print(f"Warning: No 'original' folder found in {pressure}. Skipping this folder.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  
    # 创建 DataNormalizer 对象
    normalizer = DataNormalizer()

    # 设置主文件夹路径，这里假设文件夹路径为 "data"
    parent_folder = '../data'

    # 执行数据归一化处理
    normalizer.process_all_files(parent_folder)

        


