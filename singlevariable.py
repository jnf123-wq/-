import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def singlevariab_linear_regression(real_path, real_alpha, real_iter):
    # --------------------- Part 0: 读取数据 ---------------------
    path = real_path
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    # --------------------- Part 1: 准备数据 ---------------------
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]  # Population列
    y = data.iloc[:, cols-1:cols]  # Profit列

    X.insert(0, 'Ones', 1)  # 添加偏置项
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # 初始化theta
    theta = np.matrix([0.0, 0.0])

    # --------------------- Part 2: 成本函数 ---------------------
    def computeCost(X, y, theta):
        m = len(y)
        J = (1 / (2 * m)) * np.sum(np.square(X * theta.T - y))
        return J

    # 初始成本
    initial_cost = computeCost(X, y, theta)
    print("Initial cost:", initial_cost)

    # --------------------- Part 3: 梯度下降 ---------------------
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

    # 训练
    alpha = real_alpha
    iters = real_iter
    g, cost = gradientDescent(X, y, theta, alpha, iters)

    # 最终成本
    final_cost = computeCost(X, y, g)
    print("Final cost:", final_cost)
    print("Optimized parameters:", g)

    # --------------------- Part 4: 可视化结果 ---------------------
    # 拟合直线
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + g[0, 1] * x  # θ₀ + θ₁*x

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc='best')
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    ax.grid(True)
    plt.show()

    # 成本下降曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(len(cost)), cost, 'b')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Reduction Over Iterations')
    ax.grid(True)
    plt.show()

    return g, cost
