import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 第一部分：生成5x5单位矩阵 ===
def warmUpExercise():
    return np.eye(5)

print("运行 warmUpExercise...\n", warmUpExercise())

# === 第二部分：数据可视化 ===
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]  # 人口（单位：万）
y = data[:, 1]  # 利润（单位：万美元）
m = len(y)

def plotData(X, y):
    plt.scatter(X, y, c='red', marker='x', label='Training Data')
    plt.xlabel('Population (10,000s)')
    plt.ylabel('Profit ($10,000s)')
    plt.legend()
    plt.show()

plotData(X, y)

# === 第三部分：成本函数和梯度下降 ===
X = np.column_stack((np.ones(m), X))  # 添加截距项
theta = np.zeros(2)
alpha = 0.01
iterations = 1500

def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqErrors = (predictions - y) ** 2
    return 1 / (2 * m) * np.sum(sqErrors)

# 初始成本计算
print("初始成本:", computeCost(X, y, theta))  # 应输出≈32.07

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    for _ in range(iterations):
        theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

theta, _ = gradientDescent(X, y, theta, alpha, iterations)
print("梯度下降结果:", theta)  # 应输出≈[-3.63, 1.16]

# 预测新数据
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
print("人口3.5万预测利润:", predict1 * 10000, "$")
print("人口7万预测利润:", predict2 * 10000, "$")

# === 第四部分：可视化J(θ) ===
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, y, t)

# 3D表面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_grid, theta1_grid, J_vals.T, cmap='viridis')
ax.set_xlabel('θ0')
ax.set_ylabel('θ1')
ax.set_zlabel('J(θ)')
plt.show()

# 等高线图
plt.contour(theta0_grid, theta1_grid, J_vals.T, levels=np.logspace(-2, 3, 20))
plt.scatter(theta[0], theta[1], c='red', marker='x')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.show()