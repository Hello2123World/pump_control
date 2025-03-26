import pandas as pd

# 假设你已经加载了一个DataFrame，命名为 df
df = pd.read_csv('target_0.42_训练集.csv')

# 遍历每一行
for index, row in df.iterrows():
    # 获取当前行的四个泵功率
    pump_power = [row['pump_1_power'], row['pump_2_power'], row['pump_3_power'], row['pump_4_power']]

    # 过滤出大于 1 的两个值和小于等于 1 的两个值
    greater_than_1 = [x for x in pump_power if x > 1]
    less_than_or_equal_to_1 = [x for x in pump_power if x <= 1]

    # 将大于 1 的两个数放到 pump_1_power 和 pump_2_power
    df.at[index, 'pump_1_power'] = greater_than_1[0]
    df.at[index, 'pump_2_power'] = greater_than_1[1]

    # 如果找到的值小于等于 1 的数量小于 2，填充默认值
    if len(less_than_or_equal_to_1) < 2:
        print('缺失行为:',row)
        # 对于小于等于1的值，填充缺失的部分为0
        less_than_or_equal_to_1 += [0] * (2 - len(less_than_or_equal_to_1))

    # 将小于等于 1 的两个数放到 pump_3_power 和 pump_4_power
    df.at[index, 'pump_3_power'] = less_than_or_equal_to_1[0]
    df.at[index, 'pump_4_power'] = less_than_or_equal_to_1[1]

# 打印结果
print(df.head(100))

# 保存为新的数据集
df.to_csv('target_train.csv', index=False)
# 打印结果
print('已经保存到 processed_dataset.csv')