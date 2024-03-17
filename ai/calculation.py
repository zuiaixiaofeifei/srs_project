import numpy as np
from kinematics import Kinematics
def polar_to_cartesian(r, theta):
    """
    将极坐标转换为笛卡尔坐标。

    参数:
    - r: 极径。
    - theta: 极角。

    返回:
    - x, y: 笛卡尔坐标。
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def calculate_polar_coordinates_from_trajectory(trajectory, link_lengths):
    """
    从轨迹信息计算每个机械臂的极坐标。

    参数:
    - trajectory: 包含每个点信息的字典列表，每个字典包含每个机械臂的 'theta'、't' 信息。
    - link_lengths: 机械臂的长度列表。

    返回:
    - polar_coordinates: 包含每个机械臂的极坐标 (r, theta) 的列表。
    """
    m = len(link_lengths)  # 机械臂数量
    polar_coordinates = []

    for point in trajectory:
        theta = point['theta']
        r = 0
        for i in range(m):
            r += link_lengths[i] * np.prod(np.cos(theta[:i+1]))

        polar_coordinates.append((r, theta[-1]))

    return polar_coordinates
'''
# 示例用法
link_lengths = [1, 1, 0.5]

# 假设有一个轨迹的例子
example_trajectory = [
    {'theta': [np.pi/4, np.pi/3, np.pi/6], 't': [0.1, 0.2, 0.1]},
    {'theta': [np.pi/3, np.pi/4, np.pi/5], 't': [0.2, 0.1, 0.2]},
    # 添加更多轨迹点...
]

# 计算极坐标
polar_coordinates_example = calculate_polar_coordinates_from_trajectory(example_trajectory, link_lengths)

# 转换为笛卡尔坐标
cartesian_coordinates_example = [polar_to_cartesian(r, theta) for r, theta in polar_coordinates_example]

# 打印结果
print("机械臂极坐标:")
print(polar_coordinates_example)
print("\n机械臂笛卡尔坐标:")
print(cartesian_coordinates_example)
'''