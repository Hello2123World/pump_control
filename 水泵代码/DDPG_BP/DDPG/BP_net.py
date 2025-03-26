import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os
'''BP网络中数据是二维的'''
class BPNetworkTrainer:
    def __init__(self, model, action_scaler, state_scaler):
        """
        初始化 BPNetworkTrainer 类
        :param model: 一个初始化好的 BP 神经网络模型
        :param action_scaler: 动作数据归一化器
        :param state_scaler: 状态数据归一化器
        """
        self.model = model
        self.action_scaler = action_scaler
        self.state_scaler = state_scaler

    def train_bp_model(self, file_path, pressure,index):
        """输入动作 -> 状态"""
        # 数据加载和预处理
        data = pd.read_csv(file_path)
        Y = data[['pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure', 'sum_pressure',
                  'target_pressure', 'flow']].values
        X = data[['pump_1_frequency', 'pump_2_frequency', 'pump_3_frequency', 'pump_4_frequency']].values



        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        # 创建优化器和损失函数
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.1)
        loss_fn = torch.nn.MSELoss()  # 均方误差损失函数

        # 训练模型
        for epoch in range(5000):
            self.model.train()
            optimizer.zero_grad()

            # 前向传播
            output = self.model(X)

            # 计算损失
            loss = loss_fn(output, Y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.5f}')


        directory = f'./bp_net/{pressure}'
        # 检查目录是否存在，如果不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"目录 {directory} 创建成功")
        else:
            print(f"目录 {directory} 已经存在")
        # 保存模型
        model_save_path = f'./bp_net/{pressure}/BP_{pressure}_{index}.pth'
        torch.save(self.model.state_dict(), model_save_path)
        print(f"模型已保存至: {model_save_path}")


class BPNNModel:
    def __init__(self, model_path, action_scaler_path, state_scaler_path):
        # 定义模型结构（需要与保存时的模型结构一致）
        self.model = self.BPNN()
        # 加载保存的模型参数
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # 设置为评估模式
        # 加载动作和状态的归一化器
        self.action_scaler = joblib.load(action_scaler_path)
        self.state_scaler = joblib.load(state_scaler_path)

    class BPNN(nn.Module):
        def __init__(self):
            super(BPNNModel.BPNN, self).__init__()
            self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
            self.fc2 = nn.Linear(10, 7)  # 隐藏层到输出层

        def forward(self, x):
            x = self.fc1(x)
            x = torch.tanh(self.fc2(x)) # 输出层使用tanh
            return x

    def BP_action(self, input_action):
        """
        输入一个归一化动作，直接输入网络，得到归一化的状态输出。
        """
        # 转换为 tensor 并送入模型
        action_tensor = torch.tensor(input_action, dtype=torch.float32)
        with torch.no_grad():
            state = self.model(action_tensor)
        # 返回归一化的状态
        return state.detach().numpy()

    def inverse_normalize_action(self, normalized_action):
        """
        输入一个归一化的动作，输出逆归一化后的原始动作。
        """
        original_action = self.action_scaler.inverse_transform(normalized_action)
        return np.array(original_action)

    def inverse_normalize_state(self, normalized_state):
        """
        输入一个归一化的状态，输出逆归一化后的原始状态。
        """
        # 将目标压力框定为-1
        normalized_state[0][5] = -1
        original_state = self.state_scaler.inverse_transform(normalized_state)
        original_state = np.array(original_state)

        return original_state

    def predict_bp_model(self, input_action):
        """
        输入一个动作，返回归一化后的状态。
        """
        # 对输入动作进行归一化
        normalized_action = self.action_scaler.transform(input_action)
        print(f'输入动作: {input_action}, 归一化的动作: {normalized_action}')

        # 获取归一化后的状态
        normalized_state = self.model(torch.tensor(normalized_action, dtype=torch.float32)).detach().numpy()
        print(f'归一化的状态: {normalized_state}')

        # 将归一化的状态逆归一化
        original_state = self.inverse_normalize_state(normalized_state)
        print(f'逆归一化后的状态: {original_state}')

        return original_state

# 使用示例
if __name__ == "__main__":
    # 设置 numpy 输出选项，取消科学计数法，保留四位小数
    np.set_printoptions(precision=4, suppress=True)
    # 输入值
    pressure = 0.45
    file_path = './data/normalized_target_0.45.csv'
    input_action = np.array([[43.43, 43.8, 0, 0.01]])

    # 文件路径
    model_path = f'./bp_net/BP_parameters_{pressure}.pth'
    action_scaler_path = f'./scaler/action_normalizer_{pressure}.pkl'
    state_scaler_path = f'./scaler/state_normalizer_{pressure}.pkl'

    # BPNN 神经网络模型
    model = BPNNModel.BPNN()
    # 加载归一化器
    action_scaler = joblib.load(action_scaler_path)
    state_scaler = joblib.load(state_scaler_path)

    # 创建 BPNetworkTrainer 类实例
    trainer = BPNetworkTrainer(model, action_scaler, state_scaler)

    # 调用训练方法
    trainer.train_bp_model(file_path, pressure)

    # 使用 BPNNModel 进行预测
    bp_model = BPNNModel(model_path=model_path,
                         action_scaler_path=action_scaler_path,
                         state_scaler_path= state_scaler_path)

    # 获取预测的原始状态
    original_state = bp_model.predict_bp_model(input_action)
    print(f'预测的原始状态: {original_state}')