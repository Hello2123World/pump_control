import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import ReplayBuffer
import torch
import pickle
import os

# 假设数据集是一个CSV文件，可以根据实际情况更改路径
data = pd.read_csv('target_train.csv')

# 初始化状态和动作的 MinMaxScaler
state_scaler = MinMaxScaler()
action_scaler = MinMaxScaler()

# 定义需要归一化的列
columns_to_normalize_state = [
    'pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure',
    'sum_pressure', 'target_pressure', 'flow'
]

columns_to_normalize_action = [
    'pump_1_power', 'pump_2_power', 'pump_3_power', 'pump_4_power'
]

# 对状态进行归一化
data[columns_to_normalize_state] = state_scaler.fit_transform(data[columns_to_normalize_state])

# 对动作进行归一化
data[columns_to_normalize_action] = action_scaler.fit_transform(data[columns_to_normalize_action])

# 保存 scaler 以便后续逆归一化使用
scalers_folder = './scalers'
os.makedirs(scalers_folder, exist_ok=True)

# 保存状态 scaler
with open(os.path.join(scalers_folder, 'state_scaler.pkl'), 'wb') as f:
    pickle.dump(state_scaler, f)

# 保存动作 scaler
with open(os.path.join(scalers_folder, 'action_scaler.pkl'), 'wb') as f:
    pickle.dump(action_scaler, f)

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化经验池
buffer = ReplayBuffer(state_dim=7, action_dim=4, device=device)

# 定义状态和动作的数量
state_dim = 7  # pump_1_pressure, pump_2_pressure, pump_3_pressure, pump_4_pressure, sum_pressure, target_pressure, flow
action_dim = 4  # pump_1_power, pump_2_power, pump_3_power, pump_4_power

# 遍历数据集并处理
for i in range(len(data) - 1):  # 最后一行数据无法获取下一个状态
    # 当前状态
    state = np.array([
        data.iloc[i]['pump_1_pressure'],
        data.iloc[i]['pump_2_pressure'],
        data.iloc[i]['pump_3_pressure'],
        data.iloc[i]['pump_4_pressure'],
        data.iloc[i]['sum_pressure'],
        data.iloc[i]['target_pressure'],
        data.iloc[i]['flow']
    ])

    # 当前动作
    action = np.array([
        data.iloc[i]['pump_1_power'],
        data.iloc[i]['pump_2_power'],
        data.iloc[i]['pump_3_power'],
        data.iloc[i]['pump_4_power']
    ])

    # 下一个状态
    next_state = np.array([
        data.iloc[i + 1]['pump_1_pressure'],
        data.iloc[i + 1]['pump_2_pressure'],
        data.iloc[i + 1]['pump_3_pressure'],
        data.iloc[i + 1]['pump_4_pressure'],
        data.iloc[i + 1]['sum_pressure'],
        data.iloc[i + 1]['target_pressure'],
        data.iloc[i + 1]['flow']
    ])

    # 奖励计算
    # 1. 实际压力和目标压力的差值的负数
    pressure_diff = abs(data.iloc[i]['sum_pressure'] - data.iloc[i]['target_pressure'])
    # 2. 四个泵频率的总和的负数
    power_sum = data.iloc[i]['pump_1_power'] + data.iloc[i]['pump_2_power'] + \
                data.iloc[i]['pump_3_power'] + data.iloc[i]['pump_4_power']

    reward = -pressure_diff - power_sum
    print("reward = ", reward)

    # done 定义为读取表格末尾
    done = (i == len(data) - 2)  # 如果是倒数第二行，表示读取结束

    # 将 (state, action, next_state, reward, done) 存入经验池
    buffer.add(state, action, next_state, reward, done)

# 保存经验池（可选，如果需要保存经验池的话）
buffer.save('../buffers/replay_buffer.pkl')
