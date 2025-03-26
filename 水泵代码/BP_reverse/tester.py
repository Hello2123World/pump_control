import numpy as np
from BP_net_reverse import BPNNModel  # 导入 BPNNModel 类

def predict_action(input_state, pressure):
    """
    根据输入状态和压力值，检测状态中的前三个数的值，确定 index，
    然后调用 BPNNModel 进行预测并输出结果。
    """
    # 检查输入状态中的前三个值是否符合要求
    first_three = input_state[0][:3]  # 提取前三个值

    # 检查条件：两个值大于 0.1，一个值小于 0.1
    if sum(x < 0.1 for x in first_three) != 1 or sum(x > 0.1 for x in first_three) != 2:
        print("输入状态错误，请重新输入")
        return None  # 如果不满足条件，结束函数

    # 检查输入状态中的前三个值，确定 index
    if input_state[0][0] < 0.1:
        index = 3
    elif input_state[0][1] < 0.1:
        index = 2
    elif input_state[0][2] < 0.1:
        index = 1
    else:
        print("输入状态的前三个值都不小于 0.1，请检查输入数据。")
        return None
    # print("index = ", index)

    # 构造模型路径和归一化器路径
    model_path = f'./bp_net/{pressure}/{pressure}_{index}_BP_parameters_reverse.pth'
    action_scaler_path = f'./scaler/{pressure}/{pressure}_{index}_action_normalizer.pkl'
    state_scaler_path = f'./scaler/{pressure}/{pressure}_{index}_state_normalizer.pkl'

    # 创建 BPNNModel 实例
    bp_model = BPNNModel(model_path=model_path,
                         action_scaler_path=action_scaler_path,
                         state_scaler_path=state_scaler_path)

    # 使用 BPNNModel 进行预测
    original_action = bp_model.predict_bp_model(input_state)
    print(f"输入状态为: {input_state}")
    print(f"预测的原始动作: {original_action[0]}")

    return original_action

# 使用示例
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    input_state_1 = np.array([[0.431,0.429,0.014,0.012,0.41,0.41,1389.47]])  # 输入状态
    input_state_3 = np.array([[0.013,0.434,0.438,0.012,0.41,0.41,2006.94]])  # 输入状态
    pressure = 0.41  # 压力值

    # 调用函数并输出结果
    predict_action(input_state_3, pressure)
