import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
class DataNormalizer:
    def __init__(self, pressure):
        """
        初始化 DataNormalizer 类
        :param pressure: 用于文件路径和归一化器保存的压力值
        """
        self.pressure = pressure

    def normalize_data(self, file_path,index):
        """
        读取文件并对数据进行归一化处理
        :param file_path: 输入数据文件路径
        """
        # 读取文件
        data = pd.read_csv(file_path)
        print("data = ", data)

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
        # 获取当前脚本所在的目录   # 使用 os.path.join 动态构建文件路径
        current_directory = os.path.dirname(os.path.abspath(__file__))
        output_file_path = os.path.join(current_directory, f'./{pressure}/{self.pressure}_{index}_normalized_target.csv')
        new_data.to_csv(output_file_path, index=False)
        print("new_data = ", new_data)

        directory = f'../scaler/{pressure}'
        # 检查目录是否存在，如果不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"目录 {directory} 创建成功")
        else:
            print(f"目录 {directory} 已经存在")

        action_scaler_path = os.path.join(current_directory,f'../scaler/{pressure}', f'{self.pressure}_{index}_action_normalizer.pkl')
        state_scaler_path = os.path.join(current_directory, f'../scaler/{pressure}',f'{self.pressure}_{index}_state_normalizer.pkl')

        joblib.dump(action_scaler, action_scaler_path)
        joblib.dump(state_scaler, state_scaler_path)

        print(f"action_scaler 已经保存至 {action_scaler_path}")
        print(f"state_scaler 已经保存至 {state_scaler_path}")

        return new_data, action_scaler, state_scaler

    def check_index(self,file_path):
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 获取泵的压力列
        pump_1_pressure = df['pump_1_pressure']
        pump_2_pressure = df['pump_2_pressure']
        pump_3_pressure = df['pump_3_pressure']

        # 初始化index
        index = None

        # 遍历每一行
        for i, (p1, p2, p3) in enumerate(zip(pump_1_pressure, pump_2_pressure, pump_3_pressure)):
            # 判断泵的压力
            if p3 < 0.1:
                new_index = 1
            elif p2 < 0.1:
                new_index = 2
            elif p1 < 0.1:
                new_index = 3
            else:
                new_index = index  # 如果没有条件成立，保持原来的index

            # 如果index值发生变化，打印错误并返回当前行号
            if index is None:
                index = new_index  # 第一次赋值
            elif index != new_index:
                print(f"错误的index，发生变化在第{i + 2}行")
                return -1  # 返回发生错误的行号

        # 如果index没有变化，打印正确的index
        print(f"正确的index: {index}")
        return index


# 使用示例
if __name__ == "__main__":
    # 设置压力值
    pressure = 0.35
    # 1,2 - 1 ; 2,3 - 2 ; 1,3 - 3
    index = 1

    # 初始化 DataNormalizer 类
    normalizer = DataNormalizer(pressure)
    # 输入文件路径，执行归一化处理
    file_path = f'./{pressure}/14_{pressure}_{index}.csv'
    check = normalizer.check_index(file_path=file_path)
    if check != -1:
        new_data, action_scaler, state_scaler = normalizer.normalize_data(file_path, index)
    else:
        print("输入数据不正确，请检查报错行数")


