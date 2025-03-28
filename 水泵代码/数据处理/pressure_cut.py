import pandas as pd
import os

def merge_pressure(target_pressure, file_path):
    # 加载数据
    df = pd.read_csv(file_path)

    # 函数: 检查是否满足读取条件
    def check_reading_condition(segment, target_pressure):
        # 计算满足条件的行数
        valid_count = ((segment['sum_pressure'] >= target_pressure - 0.003) & 
                       (segment['sum_pressure'] <= target_pressure + 0.003)).sum()
        
        
        return valid_count >= 30  # 如果满足条件，返回True，否则返回False

    # 函数: 检查退出条件
    def check_exit_condition(segment, target_pressure):
        invalid_count = ((segment['sum_pressure'] < target_pressure - 0.003) | 
                         (segment['sum_pressure'] > target_pressure + 0.003)).sum()
        return invalid_count >= 30  # 如果30行超出范围，满足退出条件

    # 函数: 检查并创建文件夹
    def create_folder(target_pressure):
        folder_name = str(target_pressure)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    new_data = []  # 用于存储符合条件的数据

    # 遍历表格，按每 60 行检查
    for i in range(0, len(df), 60):
        segment = df.iloc[i:i + 60]  # 获取当前60行数据
        
        # 如果该段数据满足读取条件，将其添加到新表格中
        if check_reading_condition(segment, target_pressure):
            print("符合要求：",segment["sum_pressure"])
            new_data.append(segment)
        
        # 如果该段数据不符合退出条件，则跳过
        if check_exit_condition(segment, target_pressure):
            # print("请检查: ", i)
            continue

    # 合并所有符合条件的数据
    if new_data:
        final_df = pd.concat(new_data, ignore_index=True)

        # 3. 处理新建表格，设置表头和 target_pressure 列
        final_df['target_pressure'] = target_pressure  # 添加目标压力列
        final_df = final_df.rename(columns={col: col for col in df.columns})  # 保持表头一致

        # 4. 创建文件夹并保存文件
        folder_path = create_folder(target_pressure)  
        file_name = "sum.csv"
        if file_name:
            final_df.to_csv(os.path.join(folder_path, file_name), index=False)
            print(f"文件已保存到文件夹 '{folder_path}' 下，文件名为 '{file_name}'")
            return f"./{folder_path}/{file_name}"
    else:
        print("没有符合条件的数据。")


# 函数: 按照需求处理数据并分割保存
def split(file_path,target_pressure):
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 1. 新建 'index' 列，根据各个压力值进行设置
    df['index'] = 0  # 初始化 index 列
    df.loc[df['pump_3_pressure'] < 0.1, 'index'] = 1
    df.loc[df['pump_2_pressure'] < 0.1, 'index'] = 3
    df.loc[df['pump_1_pressure'] < 0.1, 'index'] = 2
    
    # 2. 根据 'index' 列的值分割数据并保存
    for index_value in [1, 2, 3]:
        # 获取对应 index 的数据
        filtered_df = df[df['index'] == index_value]
        # 删除 'index' 和 'time' 列
        filtered_df = filtered_df.drop(columns=['index', 'time'])
        # 生成文件名，如 1.csv, 2.csv, 3.csv
        file_name = f"./{target_pressure}/{index_value}.csv"
        
        # 保存到文件夹
        filtered_df.to_csv(file_name, index=False)
        print(f"文件 '{file_name}' 已保存.")

if __name__ == "__main__":
    # 调用函数，传入参数 target_pressure 和 file_path
    target_pressure = 0.45
    file_path = './good-data/sum.csv'
    merge_pressure_path = merge_pressure(target_pressure, file_path)
    # 按照序号分割数据
    split(merge_pressure_path,target_pressure)



