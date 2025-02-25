import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
# 时间步长
dt = 1.0
# 初始状态 [x, y, vx, vy]，x和y是位置，vx和vy是速度
x0 = np.array([0, 0, 1, 1]).reshape(-1, 1)
# 初始估计误差协方差矩阵
P0 = np.eye(4) * 100
# 状态转移矩阵 F
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
# 过程噪声协方差矩阵 Q
q = 0.1
Q = np.eye(4) * q
# 观测矩阵 H，只观测位置信息
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
# 观测噪声协方差矩阵 R
r = 1.0
R = np.eye(2) * r

# 模拟地标运动和观测
num_steps = 50
true_states = [x0]
observations = []
for _ in range(num_steps):
    # 生成真实状态
    w = np.random.multivariate_normal([0, 0, 0, 0], Q).reshape(-1, 1)
    new_state = F @ true_states[-1] + w
    true_states.append(new_state)
    # 生成观测值
    v = np.random.multivariate_normal([0, 0], R).reshape(-1, 1)
    observation = H @ new_state + v
    observations.append(observation)

# 卡尔曼滤波
estimated_states = [x0]
P = P0
for obs in observations:
    # 预测步骤
    x_pred = F @ estimated_states[-1]
    P_pred = F @ P @ F.T + Q
    # 更新步骤
    y = obs - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred
    estimated_states.append(x_est)

# 提取真实位置、观测位置和估计位置
true_positions = np.array([state[:2].flatten() for state in true_states])
observed_positions = np.array([obs.flatten() for obs in observations])
estimated_positions = np.array([state[:2].flatten() for state in estimated_states])

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Position')
plt.scatter(observed_positions[:, 0], observed_positions[:, 1], c='r', marker='x', label='Observed Position')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'g--', label='Estimated Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter for Landmark Position Estimation')
plt.legend()
plt.grid(True)
plt.show()