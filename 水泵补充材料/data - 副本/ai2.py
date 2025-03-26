import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import random

# Step 1: 数据读取
df = pd.read_csv('target_0.42_处理后.csv')

# 检查列名
print(df.columns)

# 数据预处理
X = df[['pump_1_pressure', 'pump_2_pressure', 'pump_3_pressure', 'pump_4_pressure', 'sum_pressure', 'flow']]
X.loc[:, 'target_pressure'] = 0.42  # target_pressure始终为0.42
y = df[['pump_1_power', 'pump_3_power']]

# 数据集划分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 数据加载
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train.values, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义神经网络模型
class PumpPowerModel(nn.Module):
    def __init__(self):
        super(PumpPowerModel, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 输出2个值
        self.sigmoid = nn.Sigmoid()

        # 设置 power 输出范围
        self.min_power = 43.0
        self.max_power = 45.0

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x) * (self.max_power - self.min_power) + self.min_power  # 将输出限制在[min_power, max_power]
        return x


# 初始化模型
model = PumpPowerModel()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    total_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader):.4f}')

# 保存模型
torch.save(model.state_dict(), 'pump_power_model_limited.pth')


# 随机输入并预测
def random_predict():
    random_index = random.randint(0, len(df) - 1)
    random_row = df.iloc[random_index]

    pump_1_pressure = random_row['pump_1_pressure']
    pump_2_pressure = random_row['pump_2_pressure']
    pump_3_pressure = random_row['pump_3_pressure']
    pump_4_pressure = random_row['pump_4_pressure']
    sum_pressure = random_row['sum_pressure']
    flow = random_row['flow']

    print(f"随机选取的输入参数: \n"
          f"pump_1_pressure: {pump_1_pressure}, pump_2_pressure: {pump_2_pressure}, "
          f"pump_3_pressure: {pump_3_pressure}, pump_4_pressure: {pump_4_pressure}, "
          f"sum_pressure: {sum_pressure}, flow: {flow}, target_pressure: 0.42")

    input_data = [[pump_1_pressure, pump_2_pressure, pump_3_pressure, pump_4_pressure, sum_pressure, flow, 0.42]]
    input_scaled = scaler.transform(input_data)

    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    pump1_power_pred = prediction[0, 0].item()
    pump3_power_pred = prediction[0, 1].item()

    print(f"预测的 pump1_power: {pump1_power_pred:.2f}, pump3_power: {pump3_power_pred:.2f}")


# 调用随机预测
random_predict()
