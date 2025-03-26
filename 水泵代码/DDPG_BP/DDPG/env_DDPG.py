import numpy as np
import gym
from gym import spaces


class PumpEnvironment(gym.Env):
    """
    自定义强化学习环境，用于水泵的输出功率和目标压力训练。
    """

    def __init__(self, data,bp_model):
        """
        初始化环境。
        :param data: Pandas DataFrame，包含状态（泵的输出压力和目标压力）和动作（泵的功率）。
        """
        super(PumpEnvironment, self).__init__()

        # 数据
        self.data = data.reset_index(drop=True)  # 确保索引连续

        # 定义 observation_space：7个元素，值范围是[-1, 1]
        state_dim = 7
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

        # 定义动作空间：4个元素的数组，每个元素值范围是[-1, 1]，为连续型变量
        action_dim = 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # 初始化环境状态
        self.current_index = 0  # 当前数据索引
        self.state = None  # 当前状态

        # 初始化BP模型
        self.bp_model = bp_model
        # 初始化计数器
        self.cnt = 0

    def reset(self):
        """
        初始化环境，重置到第一个状态。
        :return: 初始化后的状态。
        """
        self.current_index = 0
        self.state = self.data.iloc[self.current_index][[
            'pump_1_pressure', 'pump_2_pressure',
            'pump_3_pressure', 'pump_4_pressure',
            'sum_pressure', 'target_pressure','flow'
        ]].values
        # print(f"state : {self.state}")
        return self.state

    def step(self, normalized_action_single):
        """
        传入一个归一化的动作，放入BP网络之后输出一个归一化的状态，将动作和状态逆归一化之后计算奖励，返回归一化的动作和状态
        注意！！BP网络中是个[[]]数组的结构，
        执行给定的动作，返回下一状态、奖励、是否结束以及附加信息。
        :param action: 动作（泵的功率组合）。
        :return: (next_state, reward, done, info)
        """
        self.cnt += 1
        np.set_printoptions(precision=4, suppress=True)
        # 处理为BP网络输入结构
        normalized_action = [normalized_action_single]

        # 通过BP网络输出归一化状态
        normalized_next_state = self.bp_model.BP_action(normalized_action)
        # 逆归一化状态之后计算奖励
        # 逆归一化状态
        # print(f"normalized_next_state: {normalized_next_state}")
        # print("normalized_next_state[0] = ", normalized_next_state[0])
        # print("normalized_next_state[0][5] shape = ",normalized_next_state[0][5].shape)
        next_state = self.bp_model.inverse_normalize_state(normalized_next_state)
        # 逆归一化动作
        action = self.bp_model.inverse_normalize_action(normalized_action)
        action = action[0]
        # 更新状态，计算奖励
        next_state = next_state[0]
        # 更新当前状态，归一化状态
        self.state = normalized_next_state[0]
        #print("env_state1 = ",self.state)
        done = False
        if done:
            normalized_next_state = self.state
            reward = 0  # 无奖励
        else:
            # 奖励函数
            # (1) 压力偏差的惩罚
            target_pressure = next_state[-2]  # 目标压力
            actual_pressure = next_state[-3]  # 实际输出压力
            pressure_penalty = -abs(target_pressure - actual_pressure) * 10

            # (2) 动作之间差距过大惩罚
            action_diff = -abs(action[0]-action[1])

            # (3) 曲线奖励
            efficiency_reward = self.calculate_reward(normalized_action_single)
            # (4) 让后两位靠近最小值
            action_min = -(action[2] + action[3])
            # 总奖励
            reward = pressure_penalty + efficiency_reward +action_diff + action_min
            if self.cnt % 200 == 0:
                print(f"pressure_penalty:{pressure_penalty} , action_diff:{action_diff} , "
                      f"efficiency_reward:{efficiency_reward} , reward:{reward}")
        #print("env_state2 = ", self.state)
        return self.state, reward, done, {}

    def calculate_reward(self,action):
        """
        输入动作，利用正态分布返回奖励值
        """
        sigma = 1
        mu = 0
        x1 = action[0]
        y1 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1 - mu) / sigma) ** 2)
        x2 = action[1]
        y2 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2 - mu) / sigma) ** 2)
        y = y1 + y2
        reward =  y * 10
        return reward



    def render(self, mode='human'):
        """
        可视化环境，输出当前状态信息。
        :param mode: 渲染模式，当前仅支持 'human'。
        """
        if mode == 'human':
            print(f"Step: {self.current_index}")
            print(f"State: {self.state}")
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented.")

    def close(self):
        """
        可选的清理方法。
        """
        pass