import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x1 = torch.relu(self.fc1(state))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        # print(f"Actor_x1:{x1[:3, :20]}")
        # print(f"Actor_x2:{x2[:3, :20]}")
        # print(f"Actor_x3:{x3[:3, :]}")
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


    def initialize_actor(self, reference_action):
        """
        初始化 Actor 网络的偏置，使其初始输出接近参考动作。
        :param reference_action: 表格中的第一个动作，作为参考动作
        """

        def initialize_actor(self, reference_action):
            """
            精确初始化 Actor 网络，使得初始输出为参考动作。
            """
            # 确保最后一层的权重为 0，偏置为参考动作的映射值
            with torch.no_grad():
                # 前两层使用随机初始化
                for layer in [self.actor.fc1, self.actor.fc2]:
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

                # 最后一层：tanh^{-1}((reference_action - action_min) / (action_max - action_min) * 2 - 1)
                # 因为 Actor 的输出是 tanh(最后一层) 后映射到 [action_min, action_max]
                scaled_reference = (reference_action - self.action_low) / (self.action_high - self.action_low) * 2 - 1
                scaled_reference = np.arctanh(scaled_reference)  # 反向计算 tanh 的输入

                # 设置最后一层的权重为 0，偏置为 scaled_reference
                nn.init.zeros_(self.actor.fc3.weight)
                self.actor.fc3.bias.data.copy_(torch.tensor(scaled_reference, dtype=torch.float32))