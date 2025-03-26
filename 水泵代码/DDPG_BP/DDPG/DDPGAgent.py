import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt
from sympy.physics.units import current

from env_DDPG import PumpEnvironment
from BP_net import BPNNModel

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.cnt = 0
    def forward(self, state):
        x1 = torch.relu(self.fc1(state))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        self.cnt += 1

        # if self.cnt % 200 == 0:
        #     print("cnt = ",self.cnt)
        #     print(f"Actor_x1:{x1[:3, :20]}")
        #     print(f"Actor_x2:{x2[:3, :20]}")
        #     print(f"Actor_x3:{x3[:3, :]}")
        action = torch.tanh(x3)
        # 输出在 [-1, 1] 范围内
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value



class DDPGAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 tau=0.005,
                 memory_size=1000000,
                 batch_size=64
                 ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize the Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim).float()
        self.actor_target = Actor(state_dim, action_dim).float()
        self.actor_target.load_state_dict(self.actor.state_dict())  # Copy the weights

        self.critic = Critic(state_dim, action_dim).float()
        self.critic_target = Critic(state_dim, action_dim).float()
        self.critic_target.load_state_dict(self.critic.state_dict())  # Copy the weights

        # Optimizers for Actor and Critic Networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Replay Buffer
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.memory_counter = 0  # 初始化记忆计数器

        # Discount factor and Soft Update parameter
        self.gamma = gamma
        self.tau = tau

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储经验到经验池
        """
        if len(self.memory) >= 5000:
            self.memory.popleft()  # 删除最早存储的记忆
        self.memory.append((state, action, reward, next_state, done))
        self.memory_counter += 1  # 更新记忆计数器

    def choose_action(self, state, noise=0.1):
        """
        Choose an action given a state (with added noise for exploration).
        """

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # 将state传入actor网络得到action
        '''问题！'''
        action = self.actor(state).squeeze(0).detach().numpy()

        # 添加噪声以鼓励探索（噪声不影响四舍五入）
        new_action = action + noise * np.random.randn(self.action_dim)
        # print(f"action = {action} , new_action = {new_action}")
        # 限制动作范围在 [-1, 1]
        new_action = np.clip(new_action, -1.0, 1.0)
        return new_action

    def learn(self):
        """
        Sample a batch from the replay buffer and perform learning (updating Actor and Critic).
        """
        if len(self.memory) < self.batch_size:
            return  # Skip if there are not enough samples

        # Sample a batch of experiences from the memory
        batch = random.sample(self.memory, self.batch_size)
        #  从batch中解包出states, actions, rewards, next_states, dones
        #  之后就可以训练
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Update Critic Network
        next_actions = self.actor_target(next_states)  # Get next actions from target actor
        target_q_values = self.critic_target(next_states, next_actions)  # Get next Q-values from target critic
        target_q_values = rewards + (self.gamma * target_q_values * (1 - dones))  # Bellman equation

        q_values = self.critic(states, actions)  # Get current Q-values from critic
        critic_loss = nn.MSELoss()(q_values, target_q_values.detach())  # MSE loss between predicted and target Q-values

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor Network
        predicted_actions = self.actor(states)  # Get predicted actions from actor
        actor_loss = -self.critic(states, predicted_actions).mean()  # Negative Q-value as loss (maximize Q-value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update the target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        return critic_loss.item(), actor_loss.item()  # 返回损失值

    def soft_update(self, local_network, target_network):
        """
        Soft update of the target networks.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, actor_path, critic_path):
        """
        保存 Actor 和 Critic 网络模型到指定路径
        """
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"模型已保存至 {actor_path} 和 {critic_path}")

    def train_agent(self, file_path, pressure,index, n_epochs=20,
                    max_steps_per_epoch=100, action_noise_scale=0.1,
                    noise_decay=0.99, min_noise_scale=0.01):
        """训练DDPG网络"""
        # 获取文件路径
        model_path = f'./bp_net/{pressure}/BP_{pressure}_{index}.pth'
        action_scaler_path = f'./scaler/{pressure}/action_{pressure}_{index}.pkl'
        state_scaler_path = f'./scaler/{pressure}/state_{pressure}_{index}.pkl'
        # 初始化BP模型和环境
        data = pd.read_csv(file_path)  # 输入的数据文件路径
        bp_model = BPNNModel(model_path, action_scaler_path, state_scaler_path)
        env = PumpEnvironment(data, bp_model)
        # 获取状态和动作维度
        state_dim = env.observation_space.shape[0]  # 状态维度
        action_dim = env.action_space.shape[0]  # 动作维度

        # 使用当前实例的属性初始化智能体
        self.state_dim = state_dim
        self.action_dim = action_dim

        rewards = []
        actor_losses = []
        critic_losses = []

        # 开始训练
        for epoch in range(n_epochs):
            state = env.reset()  # 初始化状态
            total_reward = 0
            steps = 0
            cnt = 0
            noise_scale = max(action_noise_scale * (noise_decay ** epoch), min_noise_scale)

            total_critic_loss = 0
            total_actor_loss = 0

            while True:
                # 通过agent选择动作
                action = self.choose_action(state, noise=noise_scale)
                # 与环境交互
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                self.store_transition(state, action, reward, next_state, done)
                if cnt % 50 == 0:
                    print(
                        f"action = {action} , reward = {reward.round(2)} , state = {state}, next_state = {next_state}")
                cnt += 1

                # 更新智能体
                if self.memory_counter > self.batch_size:
                    critic_loss, actor_loss = self.learn()
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

            print(f"Epoch {epoch + 1}/{n_epochs}: Total Reward: {total_reward:.2f}, "
                  f"Avg Critic Loss: {avg_critic_loss:.4f}, Avg Actor Loss: {avg_actor_loss:.4f}")

        # 保存模型
        directory = f'./DDPG_net/{pressure}'
        # 检查目录是否存在，如果不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"目录 {directory} 创建成功")
        else:
            print(f"目录 {directory} 已经存在")
        actor_path = f"./DDPG_net/{pressure}/actor_{pressure}_{index}.pth"
        critic_path = f"./DDPG_net/{pressure}/critic_{pressure}_{index}.pth"
        self.save(actor_path, critic_path)

        # 绘制训练过程中的结果
        self.plot_training_results(rewards, actor_losses, critic_losses, n_epochs)

    def plot_training_results(self, rewards, actor_losses, critic_losses, n_epochs):
        """绘制训练过程中的结果"""
        plt.figure(figsize=(10, 6))

        # 绘制reward的折线图
        plt.subplot(3, 1, 1)
        plt.plot(range(1, n_epochs + 1), rewards, marker='o', color='b', label='Total Reward')
        plt.title('Total Reward per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.xticks(np.arange(1, n_epochs + 1, step=1))

        # 绘制Actor Loss的折线图
        plt.subplot(3, 1, 2)
        plt.plot(range(1, n_epochs + 1), actor_losses, marker='o', color='r', label='Actor Loss')
        plt.title('Actor Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Actor Loss')
        plt.grid(True)
        plt.xticks(np.arange(1, n_epochs + 1, step=1))

        # 绘制Critic Loss的折线图
        plt.subplot(3, 1, 3)
        plt.plot(range(1, n_epochs + 1), critic_losses, marker='o', color='g', label='Critic Loss')
        plt.title('Critic Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Critic Loss')
        plt.grid(True)
        plt.xticks(np.arange(1, n_epochs + 1, step=1))

        plt.tight_layout()
        plt.show()
