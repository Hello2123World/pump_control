import torch
import torch.nn as nn
import joblib
import numpy as np

'''BP网络中数据是二维的'''

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
            x = torch.tanh(self.fc1(x))  # 隐藏层激活函数使用tanh
            x = self.fc2(x)  # 输出层
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


# 使用示例
if __name__ == "__main__":
    # 设置 numpy 输出选项，取消科学计数法，保留四位小数
    np.set_printoptions(precision=4, suppress=True)

    pressure = 0.35

    # 文件路径根据实际情况调整
    model_path = f'./bp_net/BP_parameters_{pressure}.pth'
    action_scaler_path = f'./scaler/action_normalizer_{pressure}.pkl'
    state_scaler_path = f'./scaler/state_normalizer_{pressure}.pkl'

    # 初始化模型
    bp_model = BPNNModel(model_path, action_scaler_path, state_scaler_path)

    # 原始输入动作数据
    input_action = np.array([[42.49,42.82,0.02,0.01]])
    normalized_action = bp_model.action_scaler.transform(input_action)
    # 获取归一化后的状态（BP_action 不进行逆归一化）
    normalized_state = bp_model.BP_action(normalized_action)
    print(f'输入动作: {input_action},归一化的动作: {normalized_action}\n归一化状态: {normalized_state}')

    # 将归一化的动作逆归一化
    # 这里使用 action_scaler 对原始动作归一化后的结果进行逆归一化

    original_action = bp_model.inverse_normalize_action(normalized_action)
    print(f'逆归一化后的动作: {original_action}')

    # 将归一化的状态逆归一化
    original_state = bp_model.inverse_normalize_state(normalized_state)
    print(f'逆归一化后的状态: {original_state}')
    print(f'逆归一化后的状态的类型: {type(original_state)}')
    print(f'逆归一化后的状态[0]: {original_state[0]}')