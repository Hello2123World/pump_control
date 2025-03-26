import threading
import random
import torch
import numpy as np
import pandas as pd
import json
import time
from pymodbus.client.sync import ModbusSerialClient as ModbusClient
from pymodbus.constants import Defaults
from pymodbus.payload import BinaryPayloadBuilder
import warnings
import paho.mqtt.client as mqtt
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

# BROKER 配置信息
BROKER_ADDRESS = "mqtt.czak.net"
client_id = 'mqttjs_0dfde08cb7'
BROKER_PORT = 1883
TOPIC = "/sys/OG583LL0724102801474/up"

# 初始化DataFrame
# 初始化DataFrame
df = pd.DataFrame(columns=["Timestamp", "GB2S_PL","GB3S_PL","GB4S_PL","GB1S_Par", "GB2S_Par", "GB3S_Par", "GB4S_Par", "CS_Par", "Target_Pressure", "cs_flow"])
excel_path = "./realtime_data.csv"
# 创建事件
inference_event = threading.Event()
write_event = threading.Event()
end_event = threading.Event()

# 全局变量
o_np = None
mqtt_data = None

# MQTT 连接
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("成功连接到 MQTT Broker")
        client.subscribe(TOPIC)
    else:
        print(f"连接失败，返回码: {rc}")

def on_message(client, userdata, msg):
    global mqtt_data
    print(f"收到消息，主题: {msg.topic}")
    try:
        message = msg.payload.decode("utf-8")
        data = json.loads(message)
        dev_list = data.get("devList", [])
        if not dev_list:
            print("devList 为空，跳过处理")
            return

        var_list = dev_list[0].get("varList", {})
        print(var_list)
        gb1s_par = var_list.get("GB1S_Par", 0.0)
        gb2s_par = var_list.get("GB2S_Par", 0.0)
        gb3s_par = var_list.get("GB3S_Par", 0.0)
        gb4s_par = var_list.get("GB4S_Par", 0.0)
        cs_par = var_list.get("CS_Par", 0.0)
        gb2s_pl = var_list.get("GB2S_PL", 0.0)
        gb3s_pl = var_list.get("GB3S_PL", 0.0)
        gb4s_pl = var_list.get("GB4S_PL", 0.0)
        current_hour = time.localtime().tm_hour
        if 0 <= current_hour < 4.5:
            target_pressure = 0.28
        if 4.5 <= current_hour < 5:
            target_pressure = 0.36
        if 5 <= current_hour < 5.5:
            target_pressure = 0.40
        if 5.5 <= current_hour < 8.5:
            target_pressure = 0.45
        if 8.5 <= current_hour < 12:
            target_pressure = 0.42
        if 12 <= current_hour < 13:
            target_pressure = 0.37
        if 13 <= current_hour < 15.5:
            target_pressure = 0.35
        if 15.5 <= current_hour < 22:
            target_pressure = 0.41
        if 22 <= current_hour < 23:
            target_pressure = 0.37
        if 23 <= current_hour < 24:
            target_pressure = 0.35
        cs_flow = var_list.get("CS_FLOW", 0.0)
        mqtt_data = torch.tensor([[gb2s_pl, gb3s_pl, gb4s_pl,gb1s_par, gb2s_par, gb3s_par, gb4s_par, cs_par, target_pressure, cs_flow]], dtype=torch.float32)

        print(f"生成的实时 Tensor: {mqtt_data}")

        inference_event.set()

    except json.JSONDecodeError as e:
        print(f"JSON 解码失败: {e}")
    except Exception as e:
        print(f"处理消息时发生错误: {e}")

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_start()

# 模拟接收数据并更新全局变量
def simulate_data_from_485():
    global o_np, mqtt_data
    while True:
        if mqtt_data is not None:
            o_np = mqtt_data.numpy().astype(np.float32)
            mqtt_data = None  # 清空数据
            print(f"更新 o_np: {o_np}")

            # 获取当前时间戳
            current_time = datetime.datetime.now()

            # 将数据和时间戳添加到DataFrame
            # 注意：这里假设 o_np 是一个列表或一维数组，其长度与 DataFrame 的列数匹配
            data_with_timestamp = [current_time] + o_np[0].tolist()
            df.loc[len(df)] = data_with_timestamp

            # 保存DataFrame到Excel文件
            df.to_csv(excel_path, index=False)

# 主循环
def main_loop():
    start_mqtt()
    threading.Thread(target=simulate_data_from_485).start()

# 启动主程序
if __name__ == "__main__":
    main_loop()
