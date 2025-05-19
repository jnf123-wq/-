import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def multivariable_linear_regression(file_path, real_alpha, real_iters):
    """
    对多变量线性回归数据进行训练，绘制误差下降曲线。

    Args:
        file_path (str): 数据文件路径
        alpha (float): 学习率
        iters (int): 迭代次数

    Returns:
        theta (matrix): 训练后的参数
        cost_history (ndarray): 每次迭代的损失
    """

    # 辅助函数：计算成本
    alpha=real_alpha
    iters=real_iters
    def computeCost(X, y, theta):
        m = len(y)
        J = (1 / (2 * m)) * np.sum(np.square(X * theta.T - y))
        return J

    # 辅助函数：梯度下降
    def gradientDescent(X, y, theta, alpha, iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = theta.shape[1]
        cost = np.zeros(iters)

        for i in range(iters):
            error = X * theta.T - y
            for j in range(parameters):
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)

            theta = temp
            cost[i] = computeCost(X, y, theta)

        return theta, cost

    # ----------------- Step1: 读取数据 -----------------
    data = pd.read_csv(file_path, header=None)
    data.columns = ['Feature' + str(i) for i in range(1, data.shape[1])]  # 自动命名列名
    print("First 5 rows of original data:")
    print(data.head())

    # ----------------- Step2: 特征归一化 -----------------
    data = (data - data.mean()) / data.std()

    # ----------------- Step3: 数据预处理 -----------------
    data.insert(0, 'Ones', 1)  # 添加一列1作为偏置项
    cols = data.shape[1]
    X = np.matrix(data.iloc[:, 0:cols - 1].values)
    y = np.matrix(data.iloc[:, cols - 1:cols].values)
    theta = np.matrix(np.zeros(X.shape[1]))

    # ----------------- Step4: 训练模型 -----------------
    theta, cost = gradientDescent(X, y, theta, alpha, iters)

    # ----------------- Step5: 可视化误差下降 -----------------
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.grid(True)
    plt.show()

    # ----------------- Step6: 输出最终参数 -----------------
    final_cost = computeCost(X, y, theta)
    print(f"\nFinal trained parameters:\n{theta}")
    print(f"Final cost after {iters} iterations: {final_cost:.4f}")

    return theta, cost
