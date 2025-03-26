import csv
from datetime import datetime
import pandas as pd

# 指定 CSV 文件路径
date = 9
file_path = f'./flow/flow_0{date}.csv'

# 使用 pandas 读取 CSV 文件
df = pd.read_csv(file_path)

# 查看读取结果
print(df)

# 确保 'time' 列是 datetime 类型
df['time'] = pd.to_datetime(df['time'])

# 格式化 'time' 列为 '年/月/日 时:分' 的格式，精确到分钟，省略秒
df['time'] = df['time'].dt.strftime('%Y/%m/%d %H:%M')

# 打印处理后的 DataFrame
print(df)

# 指定新的 CSV 文件路径
output_file_path = f'./flow_manage/flow_0{date}.csv'

# 将处理后的 DataFrame 保存到新的 CSV 文件
df.to_csv(output_file_path, index=False)