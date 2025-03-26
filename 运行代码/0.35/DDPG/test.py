import numpy as np
import torch
from DDPGAgent import DDPGAgent
from BP_net import BPNNModel

np.set_printoptions(precision=4, suppress=True)

def index_value(test_state):
    index = [-1, -1]
    j = 0
    for i in range(0, 4):
        if j == 2:
            break
        if test_state[i] > 0.1:
            index[j] = i
            j += 1
    return index
def swap_values(test_state,index):
    arr = test_state.copy()
    arr[index[0]], arr[0] = arr[0], arr[index[0]]
    arr[index[1]], arr[1] = arr[1], arr[index[1]]
    return arr

def inverse_value(out_put,index):
    arr = out_put.copy()
    j = 2
    arr[index[0]] = out_put[0]
    arr[index[1]] = out_put[1]
    for i in range(4):
        if i !=index[0] and i != index[1]:
            arr[i] = out_put[j]
            j += 1
    return arr

# 测试压力值
pressure = 0.35
# 测试输入状态
test_state = np.array([0.374  ,  0.376,    0.012  ,  0.01 ,    0.35  ,   0.35 , 2470. ])
index = index_value(test_state)
input_state = swap_values(test_state,index)

# 加载训练好的模型
actor_path = f"./DDPG_net/actor_{pressure}.pth"
critic_path = f"./DDPG_net/critic_{pressure}.pth"
model_path = f'bp_net/BP_parameters_{pressure}.pth'
action_scaler_path = f'./scaler/action_normalizer_{pressure}.pkl'
state_scaler_path = f'./scaler/state_normalizer_{pressure}.pkl'

# 初始化BP模型
bp_model = BPNNModel(model_path, action_scaler_path, state_scaler_path)

# 创建DDPG智能体
state_dim = 7
action_dim = 4
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.005,
    memory_size=1000000,
    batch_size=64,
)

# 加载保存的Actor网络
agent.actor.load_state_dict(torch.load(actor_path))
agent.actor.eval()  # 设置为评估模式

# 使用BP网络对状态进行归一化
normalized_state = bp_model.state_scaler.transform(input_state.reshape(1, -1))  # 转换为二维数组


# 将归一化后的状态输入到Actor网络中
normalized_action = agent.actor(torch.tensor(normalized_state, dtype=torch.float32)).detach().numpy()

# 逆归一化输出的状态
action = bp_model.inverse_normalize_action(normalized_action)
out_put = action[0]


result = inverse_value(out_put,index)


# 输出归一化和逆归一化后的状态
print(f"输入的状态: {test_state} ")
print(f"逆归一化后的动作: {result}")
print(f"归一化之后的状态:{normalized_state}")
print(f"归一化后的动作: {normalized_action}")

