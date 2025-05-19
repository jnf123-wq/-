import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_learning_rates(file_path, learning_rates, num_iters):
    """
    读取数据，比较不同学习率下的梯度下降效果，绘制结果并选出最佳学习率。

    Args:
        file_path (str): 数据文件路径
        learning_rates (list of float): 要测试的学习率列表
        num_iters (int): 每个学习率的迭代次数

    Returns:
        best_alpha (float): 最优学习率
        best_theta (matrix): 对应的最优参数
    """

    # --------------------- Part 1: 读取数据 ---------------------
    data = pd.read_csv(file_path, header=None, names=['Population', 'Profit'])

    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    X.insert(0, 'Ones', 1)
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # --------------------- Part 2: 定义辅助函数 ---------------------
    def computeCost(X, y, theta):
        m = len(y)
        J = (1 / (2 * m)) * np.sum(np.square(X * theta.T - y))
        return J

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

    # --------------------- Part 3: 训练不同学习率 ---------------------
    all_theta = []
    all_costs = []
    final_costs = []

    for alpha in learning_rates:
        theta_init = np.matrix([0.0, 0.0])
        theta, cost = gradientDescent(X, y, theta_init, alpha, num_iters)
        all_theta.append(theta)
        all_costs.append(cost)
        final_cost = computeCost(X, y, theta)
        final_costs.append(final_cost)
        print(f"Learning rate α={alpha}: Final Cost={final_cost:.4f}")

    # --------------------- Part 4: 可视化结果 ---------------------

    # 4.1 绘制拟合直线
    x_plot = np.linspace(data.Population.min(), data.Population.max(), 100)
    plt.figure(figsize=(12, 8))

    colors = ['r', 'g', 'b', 'm', 'c', 'y']  # 如果学习率超过6个，请扩展这个列表
    for idx, alpha in enumerate(learning_rates):
        f = all_theta[idx][0, 0] + all_theta[idx][0, 1] * x_plot
        plt.plot(x_plot, f, color=colors[idx % len(colors)], label=f'α={alpha}', linewidth=2)

    plt.scatter(data.Population, data.Profit, c='k', label='Training Data', s=50)
    plt.xlabel('Population (10,000s)', fontsize=12)
    plt.ylabel('Profit ($10,000s)', fontsize=12)
    plt.title('Prediction Results with Different Learning Rates', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 4.2 绘制成本函数下降曲线
    plt.figure(figsize=(12, 8))

    for idx, alpha in enumerate(learning_rates):
        plt.plot(np.arange(num_iters), all_costs[idx], color=colors[idx % len(colors)], label=f'α={alpha}', linewidth=2)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Reduction Over Iterations', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # --------------------- Part 5: 总结结果 ---------------------
    print("\nOptimization Results Summary:")
    for idx, alpha in enumerate(learning_rates):
        print(
            f"α={alpha}: Final θ0={all_theta[idx][0, 0]:.4f}, θ1={all_theta[idx][0, 1]:.4f}, Cost={final_costs[idx]:.4f}")

    best_idx = np.argmin(final_costs)
    best_alpha = learning_rates[best_idx]
    best_theta = all_theta[best_idx]
    print(f"\nBest learning rate: α={best_alpha} with cost={final_costs[best_idx]:.4f}")

    return best_alpha, best_theta
