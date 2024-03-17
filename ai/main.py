import numpy as np
import matplotlib.pyplot as plt

import optimization as op
import calculation as ca
import kinematics1 as ik
import trajectory1 as tr
import pathlength as pa



if __name__ == "__main__":
    # 定义问题参数

    target_position = [2, 1]  # 初始目标点
    link_lengths = [1, 1, 0.5]
    joint_angles = np.radians([90, 0, 0])  # 初始关节角度

    n = len(link_lengths)  # 移动机械臂自由度


    beta = np.array([1, 1])  # 权重系数


    best_individual, best_individual_fitness =  op.run ()

    trajectory_new = op.decode_individual(best_individual, n)
    polar_coordinates_last = ca.calculate_polar_coordinates_from_trajectory(trajectory_new, link_lengths)
    # 转换为笛卡尔坐标

    cartesian_coordinates_last1 = [ca.polar_to_cartesian(r, theta) for r, theta in polar_coordinates_last]
    # 根据x坐标对坐标进行排序
    cartesian_coordinates_last = sorted(cartesian_coordinates_last1, key=lambda x: x[0])

    x_first, y_first = ik.forward_kinematics(link_lengths, joint_angles)
    cartesian_coordinates_last.insert(0, (x_first, y_first))
    x_end, y_end = target_position
    cartesian_coordinates_last.append((x_end, y_end))
    x_data = [coord[0] for coord in cartesian_coordinates_last]
    y_data = [coord[1] for coord in cartesian_coordinates_last]


    print(trajectory_new)
    print(cartesian_coordinates_last)
    print("Final Result:")
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_individual_fitness)
    print("x",x_data)
    print("y",y_data)


    x_new_data, y_new_data, coefficients = tr.cubic_spline_interpolation(x_data, y_data, new_x_values = np.arange(0, 2.1, 0.1))

    #total_length = pa.cubic_spline_length(coefficients, x_data, y_data)
    #print("total_length",total_length)
    # 绘制原始数据点和拟合曲线
    plt.scatter(x_data, y_data, label='Data Points')
    plt.plot(x_new_data, y_new_data, label='Cubic Spline Fit')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Cubic Spline Interpolation')
    plt.show()






'''
    #构建轨迹点
    new_cartesian_coordinates = np.zeros(4)
    new_cartesian_coordinates[0] = ik.forward_kinematics(link_lengths, joint_angles)
    new_cartesian_coordinates[1:] = cartesian_coordinates_last
    #new_cartesian_coordinates[4] = target_position

    print(new_cartesian_coordinates)
'''






