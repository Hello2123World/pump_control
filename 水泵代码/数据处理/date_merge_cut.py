import pandas as pd
import os


def merge_sort_and_split_data(folder_path, store_path):
    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print("csv_files: ", csv_files)

    # 1. 合并所有 CSV 文件
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        all_data.append(data)

    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)

    # 保存为 sum.csv
    merged_df.to_csv(os.path.join(store_path, 'sum.csv'), index=False)

    # 2. 按日期排序
    merged_df['time'] = pd.to_datetime(merged_df['time'], errors='coerce')  # 将无法解析的日期设置为 NaT
    merged_df = merged_df.dropna(subset=['time'])  # 删除 'time' 列为 NaT 的行
    merged_df = merged_df.sort_values(by='time')  # 按照时间列排序，确保从以前到现在

    # 3. 获取所有不同的日期
    unique_dates = merged_df['time'].dt.date.unique()

    # 4. 保存每一天的数据
    for date in unique_dates:
        daily_df = merged_df[merged_df['time'].dt.date == date]
        file_name = f"{date.month}.{date.day}.csv"  # 命名为月份.日期
        daily_df.to_csv(os.path.join(store_path, file_name), index=False)

    # 输出合并后的 DataFrame 和保存的信息
    print(f"合并后的数据已保存为 sum.csv, 存储路径：{os.path.join(store_path, 'sum.csv')}")
    print(f"按日期分割后的文件已保存：")
    for date in unique_dates:
        print(f"{date.month}.{date.day}.csv")


if __name__ == "__main__":
    folder_path = './data'  # CSV 文件所在文件夹路径
    store_path = './good-data'  # 保存文件的文件夹路径
    merge_sort_and_split_data(folder_path, store_path)