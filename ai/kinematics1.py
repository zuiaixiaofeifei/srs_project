
import numpy as np
from scipy.optimize import minimize

def forward_kinematics(link_lengths, joint_angles):
    if len(link_lengths) != 3 or len(joint_angles) != 3:
        raise ValueError("The number of link lengths and joint angles should be 3.")

    l1, l2, l3 = link_lengths
    theta1, theta2, theta3 = joint_angles

    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)

    return x, y


def inverse_kinematics(target_pos, link_lengths):
    def objective_function(joint_angles):
        x, y = forward_kinematics(link_lengths, joint_angles)
        return (x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2

    # 初始猜测的关节角度
    initial_guess = np.zeros(3)

    # 使用优化算法求解逆运动学问题
    result = minimize(objective_function, initial_guess, method='L-BFGS-B')

    # 返回关节角度
    return result.x