import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os

'''BP网络中数据是二维的'''
class BPNetworkTrainer:
    def __init__(self, model):
        """
        初始化 BPNetworkTrainer 类
        :param model: 一个初始化好的 BP 神经网络模型
        :param action_scaler: 动作数据归一化器
        :param state_scaler: 状态数据归一化器
        """
        self.model = model

    def train_bp_model(self, file_path, pressure,index):
        """输入状态X -> 动作Y"""
        # 数据加载和预处理
        data = pd.read_csv(file_path)
        X = data[['pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure', 'sum_pressure',
                  'target_pressure', 'flow']].values
        Y = data[['pump_1_frequency', 'pump_2_frequency', 'pump_3_frequency', 'pump_4_frequency']].values



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

        # 保存模型
        directory = f'./bp_net/{pressure}'
        # 检查目录是否存在，如果不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"目录 {directory} 创建成功")
        else:
            print(f"目录 {directory} 已经存在")
            
        model_save_path = f'./bp_net/{pressure}/{index}_BP.pth'
        torch.save(self.model.state_dict(), model_save_path)
        print(f"模型已保存至: {model_save_path}")
        return model_save_path


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
            self.fc1 = nn.Linear(7, 10)  # 输入层到隐藏层
            self.fc2 = nn.Linear(10, 4)  # 隐藏层到输出层

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

    def predict_bp_model(self, input_state):
        """
        输入一个状态，返回归一化后的动作。
        """
        # 对输入状态进行归一化
        normalized_state = self.state_scaler.transform(input_state)
        # print(f'输入状态: {input_state}')
        # print(f"归一化的动作: {normalized_state}")

        # 获取归一化后的动作
        normalized_action= self.model(torch.tensor(normalized_state, dtype=torch.float32)).detach().numpy()
        # print(f'模型输出归一化的动作: {normalized_action}')

        # 将归一化的动作逆归一化
        original_action = self.inverse_normalize_action(normalized_action)
        # print(f'逆归一化后的动作: {original_action}')

        return original_action

def process_and_predict(input_file, output_file, model_path, action_scaler_path, state_scaler_path):
    """
    处理输入文件，使用BP网络进行预测，并将预测结果与实际目标输出保存到新的CSV文件中
    """
    # 加载数据
    data = pd.read_csv(input_file)
    X = data[['pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure', 'sum_pressure',
              'target_pressure', 'flow']].values[:]  # 读取前100行
    Y = data[['pump_1_frequency', 'pump_2_frequency', 'pump_3_frequency', 'pump_4_frequency']].values[:]
    # 初始化模型
    bp_model = BPNNModel(model_path=model_path,
                         action_scaler_path=action_scaler_path,
                         state_scaler_path=state_scaler_path)

    # 预测并保存结果
    results = []
    for i in range(X.shape[0]):
        test_state = X[i]  # 取出一行作为输入
        # print("X = ",X[i])
        original_action = bp_model.predict_bp_model([test_state])  # 预测动作
        original_action = original_action[0]
        # print("original_action = ",original_action)
        y = Y[i]
        # print("y = ", y )
        result = np.hstack((original_action, y))  # 将预测的动作和目标输出拼接
        results.append(result)

    # 转换为DataFrame并保存到新的CSV文件中
    result_df = pd.DataFrame(results, columns=['pred_pump_1_freq', 'pred_pump_2_freq', 'pred_pump_3_freq', 'pred_pump_4_freq',
                                               'actual_pump_1_freq', 'actual_pump_2_freq', 'actual_pump_3_freq', 'actual_pump_4_freq'])
    # 使用 round() 保留四位小数
    result_df = result_df.round(2)
    result_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 设置 numpy 输出选项，取消科学计数法，保留四位小数
    np.set_printoptions(precision=4, suppress=True)
    # 保证运行时候脚本位置
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  
    
    # 自动遍历所有压力值和序号
    data_dir = './data'
    # 获取所有压力值文件夹
    pressure_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    
    for pressure_folder in pressure_folders:
        pressure = float(pressure_folder)  # 将文件夹名转换为浮点数
        pressure_path = os.path.join(data_dir, pressure_folder)
        
        # 获取该压力值下的所有序号文件（查找格式为 "{index}_normalized.csv" 的文件）
        index_files = [f for f in os.listdir(pressure_path) if f.endswith('_normalized.csv')]
        
        for index_file in index_files:
            # 从文件名中提取序号
            index = int(index_file.split('_')[0])
            print(f"\n开始处理: 压力值 = {pressure}, 序号 = {index}")
            
            # 设置文件路径
            file_path = f'./data/{pressure}/{index}_normalized.csv'
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}，跳过")
                continue
                
            # 设置其他路径
            action_scaler_path = f'./scaler/{pressure}/{index}_action.pkl'
            state_scaler_path = f'./scaler/{pressure}/{index}_state.pkl'
            
            # 检查归一化器文件是否存在
            if not os.path.exists(action_scaler_path) or not os.path.exists(state_scaler_path):
                print(f"归一化器文件不存在: {action_scaler_path} 或 {state_scaler_path}，跳过")
                continue
            
            try:
                # 输入示例状态（使用固定示例或从数据中获取）
                input_state = np.array([[0.378, 0.377, 0.016, 0.013, 0.351, 0.35, 2658.56]])
                
                # BPNN 神经网络模型
                model = BPNNModel.BPNN()
                
                # 加载归一化器
                action_scaler = joblib.load(action_scaler_path)
                state_scaler = joblib.load(state_scaler_path)
                
                # 创建 BPNetworkTrainer 类实例,训练BP网络
                trainer = BPNetworkTrainer(model)
                model_path = trainer.train_bp_model(file_path, pressure, index)
                
                # 使用 BPNNModel 进行预测
                bp_model = BPNNModel(model_path=model_path,
                                    action_scaler_path=action_scaler_path,
                                    state_scaler_path=state_scaler_path)
                
                # 获取预测的原始状态
                original_action = bp_model.predict_bp_model(input_state)
                print(f"输入的状态为: {input_state[0]}")
                print(f"预测的原始动作: {original_action[0]}")
                print(f"压力值 {pressure}, 序号 {index} 处理完成")
                
            except Exception as e:
                print(f"处理压力值 {pressure}, 序号 {index} 时出错: {str(e)}")
                continue
    
    print("\n所有数据处理完成！")