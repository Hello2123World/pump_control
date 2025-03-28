import pandas as pd
def merge(pressure):
    # 读取两个CSV文件
    df1 = pd.read_csv('./data/0.35/14_0.35_1.csv')
    df2 = pd.read_csv('./data/0.35/20_0.35_1.csv')

    # 垂直合并两个DataFrame
    combined_df = pd.concat([df1, df2], axis=0)

    # 重置索引（可选）
    combined_df = combined_df.reset_index(drop=True)

    # 保存合并后的文件
    combined_df.to_csv('./data/0.35/sum_0.35_1.csv', index=False)

    print("文件合并完成，已保存为sum_0.35_1.csv")

def calculate(file_path):
    """
       计算两个泵频率预测值与实际值的绝对差总和
       参数：
           file_path: CSV文件路径（例如：'0.35_1_comparison_actions.csv'）
       返回：
           两个泵预测值与实际值的绝对差总和
       """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 计算每行的绝对差值并累加
    total = 0.0
    for _, row in df.iterrows():
        diff1 = abs(row['pred_pump_1_freq'] - row['actual_pump_1_freq'])
        diff2 = abs(row['pred_pump_2_freq'] - row['actual_pump_2_freq'])
        total += (diff1 + diff2)

    print(f"绝对差总和为: {total:.2f}")
    return total

if __name__ == "__main__":
    file_path = "./compare/0.35/0.35_1_comparison_actions.csv"
    result = calculate(file_path)
    print("总和为：",result)
    file_path = "./compare/0.35/0.35_sum_comparison_actions.csv"
    result = calculate(file_path)
    print("总和为：", result)