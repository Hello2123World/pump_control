import numpy as np
import pandas as pd
from env_DDPG import PumpEnvironment
from DDPGAgent import DDPGAgent
from BP_net import BPNNModel
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# 设置压力
pressure = 0.45

# 参数设置
n_epochs = 20               # 训练的总 epoch 数
max_steps_per_epoch = 100  # 每个 epoch 中的最大步数
batch_size = 64             # 批量大小
action_noise_scale = 0.1    # 动作噪声的初始比例
noise_decay = 0.99          # 噪声逐渐减少的比例
min_noise_scale = 0.01      # 最小噪声比例

# 初始化环境
data = pd.read_csv(f"./data/normalized_target_{pressure}.csv")  # 替换为你的数据文件路径
# 文件路径根据实际情况调整
model_path = f'bp_net/BP_parameters_{pressure}.pth'
action_scaler_path = f'./scaler/action_normalizer_{pressure}.pkl'
state_scaler_path = f'./scaler/state_normalizer_{pressure}.pkl'

# 初始化BP模型
bp_model = BPNNModel(model_path, action_scaler_path, state_scaler_path)

env = PumpEnvironment(data,bp_model)

state_dim = env.observation_space.shape[0]  # 状态维度
action_dim = env.action_space.shape[0]      # 动作维度

# 创建存储每个Epoch的reward, Actor Loss, Critic Loss的列表
rewards = []
actor_losses = []
critic_losses = []


# 初始化智能体
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.005,
    memory_size=1000000,
    batch_size=64,
)
cnt = 0
# 开始训练
for epoch in range(n_epochs):
    state = env.reset()  # 初始化状态
    total_reward = 0
    steps = 0
    cnt = 0
    noise_scale = max(action_noise_scale * (noise_decay ** epoch), min_noise_scale)  # 衰减噪声
    total_critic_loss = 0
    total_actor_loss = 0

    while True:
        # 通过agent选择动作
        action = agent.choose_action(state, noise=noise_scale)
        # 与环境交互
        next_state, reward, done, _ = env.step(action)

        # 存储经验
        agent.store_transition(state, action, reward, next_state, done)
        if cnt % 50== 0:
            print(f"action = {action} , reward = {reward.round(2)} , state = {state}, next_state = {next_state}")
        cnt += 1
        # 更新智能体
        if agent.memory_counter> batch_size:
            # print("=========================开始学习=========================")
            # break
            critic_loss, actor_loss = agent.learn()
            if critic_loss is not None:
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss

        state = next_state  # 更新状态
        total_reward += reward
        steps += 1

        # 检查结束条件
        if done or steps >= max_steps_per_epoch:
            break

    avg_critic_loss = total_critic_loss / steps if steps > 0 else 0
    avg_actor_loss = total_actor_loss / steps if steps > 0 else 0

    # 将每个epoch的结果加入列表
    rewards.append(total_reward)
    actor_losses.append(avg_actor_loss)
    critic_losses.append(avg_critic_loss)

    # print(f"action = {action} , reward = {reward.round(2)} , state = {state}, next_state = {next_state}")
    print(f"Epoch {epoch + 1}/{n_epochs}: Total Reward: {total_reward:.2f}, Avg Critic Loss: {avg_critic_loss:.4f}, Avg Actor Loss: {avg_actor_loss:.4f}")

# # 绘制每个Epoch中的reward, actor_loss, critic_loss折线图
# plt.figure(figsize=(10, 6))
#
# # 绘制reward的折线图
# plt.subplot(3, 1, 1)
# plt.plot(range(1, n_epochs + 1), rewards, marker='o', color='b', label='Total Reward')
# plt.title('Total Reward per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Total Reward')
# plt.grid(True)
# plt.xticks(np.arange(1, n_epochs + 1, step=1))
# plt.grid(False)  # 关闭网格线
#
# # 绘制Actor Loss的折线图
# plt.subplot(3, 1, 2)
# plt.plot(range(1, n_epochs + 1), actor_losses, marker='o', color='r', label='Actor Loss')
# plt.title('Actor Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Actor Loss')
# plt.grid(True)
# plt.xticks(np.arange(1, n_epochs + 1, step=1))
# plt.grid(False)  # 关闭网格线
#
# # 绘制Critic Loss的折线图
# plt.subplot(3, 1, 3)
# plt.plot(range(1, n_epochs + 1), critic_losses, marker='o', color='g', label='Critic Loss')
# plt.title('Critic Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Critic Loss')
# plt.grid(True)
# plt.xticks(np.arange(1, n_epochs + 1, step=1))
# plt.grid(False)  # 关闭网格线
#
# # 显示图表
# plt.tight_layout()
# plt.show()

# 模型保存
actor_path = f"./DDPG_net/actor_{pressure}.pth"
critic_path = f"./DDPG_net/critic_{pressure}.pth"
agent.save(actor_path, critic_path)




