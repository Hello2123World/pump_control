import pandas as pd
# 读取CSV文件
date = 9
pressure = 0.35

target_df = pd.read_csv(f'data/{date}_target_{pressure}.csv')
flow_df = pd.read_csv(f'flow_manage/flow_0{date}.csv')

# 将时间字符串转换为datetime对象
target_df['time'] = pd.to_datetime(target_df['time'])
flow_df['time'] = pd.to_datetime(flow_df['time'])
# 修改列名
target_df.rename(columns={
    'pump_1_power': 'pump_1_frequency',
    'pump_2_power': 'pump_2_frequency',
    'pump_3_power': 'pump_3_frequency',
    'pump_4_power': 'pump_4_frequency'
}, inplace=True)

# 初始化索引
i = 0  # target_df的索引
j = 0  # flow_df的索引

# 处理数据
while i < len(target_df) and j < len(flow_df) - 1:
    ti = target_df.at[i, 'time']
    tj = flow_df.at[j, 'time']
    tj_next = flow_df.at[j + 1, 'time']

    if tj <= ti < tj_next:
        target_df.at[i, 'flow'] = flow_df.at[j, 'flow']
        i += 1
    else:
        j += 1

# 处理剩余的target_df数据
while i < len(target_df):
    ti = target_df.at[i, 'time']
    if j < len(flow_df):
        tj = flow_df.at[j, 'time']
        if ti >= tj:
            target_df.at[i, 'flow'] = flow_df.at[j, 'flow']
        else:
            target_df.at[i, 'flow'] = None  # 如果没有匹配的flow值，设置为None
    i += 1

# 调整列的顺序
target_df = target_df[['time', 'pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure',
                       'sum_pressure', 'target_pressure', 'flow', 'pump_1_frequency', 'pump_2_frequency',
                       'pump_3_frequency', 'pump_4_frequency']]

manage_target = f'data_manage/{date}_target_{pressure}_处理后.csv'
# 保存处理后的数据到新的CSV文件
target_df.to_csv(manage_target, index=False)
print(f"处理完成，结果已保存到{manage_target} ")

