import numpy as np
import joblib
import os
import pandas as pd

from test import DDPGTester
from datetime import datetime

def test_DDPG(pressure,test_state):
    np.set_printoptions(precision=4, suppress=True)

    tester = DDPGTester(pressure, test_state)
    tester.test_ddpg_agent()

def get_pressure_by_time(current_time, time_periods):
    """更具系统时间获取压力"""
    current_time_obj = datetime.strptime(current_time, '%H:%M')

    for period in time_periods:
        start_time_obj = datetime.strptime(period['start'], '%H:%M')
        end_time_obj = datetime.strptime(period['end'], '%H:%M')

        # 检查当前时间是否在时间段内
        if start_time_obj <= current_time_obj < end_time_obj:
            return period['pressure']

    # 如果当前时间不在任何时间段内，返回 None 或默认值
    return None

if __name__ == "__main__":
    # 设置 numpy 输出选项，取消科学计数法，保留四位小数
    np.set_printoptions(precision=4, suppress=True)

    # 时间段和压力设定值表
    time_periods = [
        {'start': '00:00', 'end': '04:30', 'pressure': 0.28},
        {'start': '04:30', 'end': '05:00', 'pressure': 0.36},
        {'start': '05:00', 'end': '05:30', 'pressure': 0.40},
        {'start': '05:30', 'end': '08:30', 'pressure': 0.45},
        {'start': '08:30', 'end': '12:00', 'pressure': 0.42},
        {'start': '12:00', 'end': '13:00', 'pressure': 0.37},
        {'start': '13:00', 'end': '15:30', 'pressure': 0.35},
        {'start': '15:30', 'end': '22:00', 'pressure': 0.41},
        {'start': '22:00', 'end': '23:00', 'pressure': 0.37},
        {'start': '23:00', 'end': '00:00', 'pressure': 0.35}
    ]
    # 获取当前时间（格式：'HH:MM'）
    current_time = datetime.now().strftime('%H:%M')
    print("系统时间为：",current_time)
    pressure = get_pressure_by_time(current_time, time_periods)

    if pressure is not None:
        print(f"当前时间 {current_time} 对应的压力设定值是 {pressure} MPa")
        # 输入归一化文件
        normalization_file_path = f'./data/normalized_target_{pressure}.csv'
        # 测试DDPG网络
        print("检测DDPG网络效果")
        test_state = np.array([0.432,0.431,0.012,0.013,0.41,0.41,2562])
        test_DDPG(pressure,test_state)


    else:
        print(f"当前时间 {current_time} 不在任何时间段内")



