# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt


# 定义绘制正态分布曲线的函数，带有可变的均值和标准差
def plot_normal_distribution(mu=0, sigma=0.4):
    # 定义x轴的范围
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)

    # 计算正态分布的y值
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # 绘制图形
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=f'Normal Distribution (μ={mu}, σ={sigma})', color='b')

    # 添加标题和标签
    plt.title(f'Normal Distribution with μ={mu} and σ={sigma}')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # 显示图例
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()


# 调用函数绘制不同均值和标准差的正态分布曲线
plot_normal_distribution(mu=0, sigma=1)

