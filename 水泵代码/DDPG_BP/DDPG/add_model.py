import numpy as np
import joblib
import os
import pandas as pd
from DDPG.normalization import DataNormalizer
from BP_net import BPNetworkTrainer,BPNNModel
from DDPGAgent import DDPGAgent
from env_DDPG import PumpEnvironment
from test import DDPGTester
def perform_normalization(file_path, pressure ,index ):
    """执行数据归一化操作"""
    # 创建 DataNormalizer 类的实例
    normalizer = DataNormalizer(pressure,index )

    check = normalizer.check_index(file_path=file_path)
    if check != -1:
        new_data, action_scaler, state_scaler = normalizer.normalize_data(file_path)
        print("归一化后的数据样本：")
        print(new_data.head())
    else:
        print("输入数据不正确，请检查报错行数")

def train_bp_network(file_path, pressure,index):
    """训练BP网络"""
    # 初始化模型和归一化器
    model = BPNNModel.BPNN()  # 假设 BPNN 是您定义的神经网络模型
    action_scaler = joblib.load(f'./scaler/{pressure}/action_{pressure}_{index}.pkl')  # 加载归一化器
    state_scaler = joblib.load(f'./scaler/{pressure}/state_{pressure}_{index}.pkl')

    # 创建 BPNetworkTrainer 类实例
    trainer = BPNetworkTrainer(model, action_scaler, state_scaler)

    # 调用训练方法
    trainer.train_bp_model(file_path, pressure,index)

def predict_bp_network(input_action, pressure,index):
    """测试BP网络"""
    # 使用 BPNNModel 进行预测
    bp_model = BPNNModel(model_path=f'./bp_net/{pressure}/BP_{pressure}_{index}.pth',
                         action_scaler_path=f'./scaler/{pressure}/action_{pressure}_{index}.pkl',
                         state_scaler_path=f'./scaler/{pressure}/state_{pressure}_{index}.pkl')

    # 获取预测的原始状态
    original_state = bp_model.predict_bp_model(input_action)
    print(f'预测的原始状态: {original_state}')

def train_DDPG(file_path, pressure,index, n_epochs=20, max_steps_per_epoch=100, batch_size=64,
                         action_noise_scale=0.1, noise_decay=0.99, min_noise_scale=0.01):
    """
    初始化 DDPGAgent，并训练模型，保存模型并绘制训练损失曲线。
    :param file_path: 输入的数据文件路径
    :param pressure: 用于保存模型的压力值
    :param n_epochs: 训练的总 epoch 数
    :param max_steps_per_epoch: 每个 epoch 的最大步数
    :param batch_size: 批量大小
    :param action_noise_scale: 动作噪声初始比例
    :param noise_decay: 噪声逐渐减少的比例
    :param min_noise_scale: 最小噪声比例
    """
    # 获取文件路径
    model_path = f'./bp_net/{pressure}/BP_{pressure}_{index}.pth'
    action_scaler_path = f'./scaler/{pressure}/action_{pressure}_{index}.pkl'
    state_scaler_path = f'./scaler/{pressure}/state_{pressure}_{index}.pkl'
    # 初始化BP模型和环境
    data = pd.read_csv(file_path)  # 输入的数据文件路径
    bp_model = BPNNModel(model_path, action_scaler_path, state_scaler_path)
    env = PumpEnvironment(data, bp_model)

    # 状态和动作的维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建 DDPGAgent 实例
    agent = DDPGAgent(state_dim=state_dim,
                      action_dim=action_dim,
                      learning_rate=0.001,
                      gamma=0.99,
                      tau=0.005,
                      memory_size=1000000,
                      batch_size=batch_size)

    # 训练模型
    agent.train_agent(file_path=file_path, pressure=pressure,index= index, n_epochs=n_epochs,max_steps_per_epoch=max_steps_per_epoch,
                      action_noise_scale=action_noise_scale, noise_decay=noise_decay,min_noise_scale = min_noise_scale)

def test_DDPG(pressure,test_state):
    # np.set_printoptions(precision=4, suppress=True)

    tester = DDPGTester(pressure = pressure, test_state=test_state,state_dim=7,action_dim=4)
    tester.test_ddpg_agent()

if __name__ == "__main__":
    # 设置 numpy 输出选项，取消科学计数法，保留四位小数
    np.set_printoptions(precision=4, suppress=True)
    # 设置压力值和次序
    # 开启 1,2 泵 index = 1
    # 开启 1,3 泵 index = 2
    # 开启 2,3 泵 index = 3
    pressure = 0.35
    index = 1

    # 对输入的数据进行归一化操作
    # print("进行归一化操作")
    # file_path = f'./data/{pressure}/{pressure}_{index}.csv'
    # perform_normalization(pressure=pressure,file_path=file_path ,index= index)


    # 输入归一化文件
    normalization_file_path = f'./data/{pressure}/normalized_{pressure}_{index}.csv'

    # 训练BP网络
    # print("正在训练BP网络")
    # train_bp_network(normalization_file_path, pressure,index)

    # 输入测试动作
    # input_action = np.array([[43.38,44.22,0.01,0.01]])
    # # 执行预测过程
    # print("检测BP网络效果")
    # predict_bp_network(input_action, pressure,index)

    # 调用函数初始化并训练模型
    # print("训练DDPG网络")
    # train_DDPG(file_path=normalization_file_path, pressure=pressure,index=index, n_epochs=20,
    #            max_steps_per_epoch=100, batch_size=64,action_noise_scale=0.1, noise_decay=0.99,
    #            min_noise_scale=0.01)

    # 测试DDPG网络
    print("检测DDPG网络效果")
    test_state = np.array([0.378,0.379,0.016,0.013,0.35,0.35,2627.89])
    test_DDPG(pressure,test_state)
