import numpy as np
import torch
from DDPGAgent import DDPGAgent
from BP_net import BPNNModel

class DDPGTester:
    def __init__(self, pressure,state_dim,action_dim, test_state):
        """
        初始化 DDPGTester 类。
        :param pressure: 压力值，用于加载模型和归一化器
        :param test_state: 输入的测试状态
        """
        self.pressure = pressure
        self.test_state = test_state
        # self.input_index = self.index_value(test_state)  # 获取索引
        # self.input_state = self.swap_values(test_state, self.input_index)  # 获取调整后的状态
        self.index = self.get_index()
        print("index = ",self.index)
        # 定义模型路径
        self.actor_path = f"./DDPG_net/{pressure}/actor_{pressure}_{self.index}.pth"
        self.critic_path =  f"./DDPG_net/{pressure}/critic_{pressure}_{self.index}.pth"
        self.model_path = f'./bp_net/{pressure}/BP_{pressure}_{self.index}.pth'
        self.action_scaler_path = f'./scaler/{pressure}/action_{pressure}_{self.index}.pkl'
        self.state_scaler_path = f'./scaler/{pressure}/state_{pressure}_{self.index}.pkl'

        # 初始化 BP 模型
        self.bp_model = BPNNModel(self.model_path, self.action_scaler_path, self.state_scaler_path)

        # 初始化 DDPG Agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            tau=0.005,
            memory_size=1000000,
            batch_size=64,
        )

        # 加载 Actor 模型
        self.agent.actor.load_state_dict(torch.load(self.actor_path))
        self.agent.actor.eval()  # 设置为评估模式

    def get_index(self):
    # 根据输入状态，返回index
        if len(self.test_state) != 7:
            raise ValueError("test_state should be an array with 7 elements.")

        count = sum(self.test_state[:3] < 0.1)

        if count == 3:
            print("错误的状态")
            return None
        elif count == 2:
            print("错误的状态")
            return None
        elif self.test_state[0] < 0.1:
            return 3
        elif self.test_state[1] < 0.1:
            return 2
        elif self.test_state[2] < 0.1:
            return 1
        else:
            print("未定义的输入状态")
            return -1

    # def index_value(self, test_state):
    #     """
    #     找到大于 0.1 的前两个值的索引。
    #     :param test_state: 输入状态
    #     :return: 索引列表
    #     """
    #     index = [-1, -1]
    #     j = 0
    #     for i in range(0, 4):
    #         if j == 2:
    #             break
    #         if test_state[i] > 0.1:
    #             index[j] = i
    #             j += 1
    #     return index
    #
    # def swap_values(self, test_state, index):
    #     """
    #     调整状态顺序，交换两个值。
    #     :param test_state: 输入状态
    #     :param index: 索引值
    #     :return: 调整后的状态
    #     """
    #     arr = test_state.copy()
    #     arr[index[0]], arr[0] = arr[0], arr[index[0]]
    #     arr[index[1]], arr[1] = arr[1], arr[index[1]]
    #     return arr

    def test_ddpg_agent(self):
        """
        运行 DDPG 智能体，输入状态，输出动作。
        :return: None
        """
        # 使用 BP 模型对状态进行归一化
        normalized_state = self.bp_model.state_scaler.transform(self.test_state.reshape(1, -1))  # 转换为二维数组

        # 将归一化后的状态输入到 Actor 网络中
        normalized_action = self.agent.actor(torch.tensor(normalized_state, dtype=torch.float32)).detach().numpy()

        # 逆归一化输出的动作
        action = self.bp_model.inverse_normalize_action(normalized_action)
        out_put = action[0]
        # result = self.swap_values(out_put, self.input_index)

        # 输出归一化和逆归一化后的状态
        print(f"输入的状态: {self.test_state}")
        print(f"逆归一化后的动作: {out_put}")
        print(f"归一化之后的状态: {normalized_state}")
        print(f"归一化后的动作: {normalized_action}")

# Example of how to use the class
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    pressure = 0.45
    test_state = np.array([0.472,0.475,0.014,0.466,0.45,0.45,2579])
    tester = DDPGTester(pressure, test_state)
    tester.test_ddpg_agent()
