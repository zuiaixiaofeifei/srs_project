import numpy as np

import pathlength as pa
import calculation as ca
import kinematics1 as ik
import trajectory1 as tr
# 定义问题参数


target_position = [2, 1]# 初始目标点
link_lengths = [1, 1, 0.5]
joint_angles = np.radians([90, 0, 0])# 初始关节角度
n  = len(link_lengths)    # 移动机械臂自由度

b = 2  # 起点与终点之间节点数目
beta = np.array([1, 1, 1])  # 权重系数

# 遗传算法参数 genetic algorithm parameter
population_size = 200
crossover_probability = 0.8
mutation_probability = 0.05
max_generations = 100

def trajectory(n, b, theta_lower=0, theta_upper=2*np.pi):
    # 生成随机轨迹，n是机械臂数量，b是轨迹点数量
    traj = []
    for _ in range(b):
        point = {'theta': np.random.uniform(theta_lower, theta_upper, size=n), 't': np.random.rand()}
        traj.append(point)
    return traj

def decode_individual(individual, n):
    """
    将个体解码为轨迹。

    参数:
    - individual: 编码后的个体。
    - n: 机械臂数量。

    返回:
    - trajectory: 包含每个点信息的字典列表，每个字典包含每个机械臂的 'theta'、't' 信息。
    """
    b = len(individual) // (2 * n)  # 计算轨迹点数量
    trajectory = []

    for i in range(b):
        point = {'theta': individual[n * i: n * (i + 1)],
                 't': individual[n * b + i]}
        trajectory.append(point)

    return trajectory


def encode_individual(trajectory, n):
    """
    将轨迹编码为一个个体。

    参数:
    - trajectory: 包含每个点信息的字典列表，每个字典包含每个机械臂的 'theta'、't' 信息。
    - n: 机械臂数量。

    返回:
    - individual: 编码后的个体。
    """
    individual = []

    for point in trajectory:
        # 对于每个机械臂的每个轨迹点，添加运动角度和时间
        for arm in range(n):
            # Ensure the index is within the valid range for 'theta'
            if arm < len(point['theta']):
                individual.append(point['theta'][arm])
            else:
                individual.append(0.0)  # Default value if 'theta' is not available
            individual.append(point['t'])  # 't' is a single value for the entire point

    return np.array(individual)




# 计算适应度函数

def fitness_function(individual):
    # 解码个体
    trajectory = decode_individual(individual, n)

    # Extract 'theta' and 't' values into separate arrays
    theta_values = np.array([point['theta'] for point in trajectory])
    t_values = np.array([point['t'] for point in trajectory])

    # 计算每个关节的总距离 F_q
    F_q = np.sum(np.abs(np.diff(theta_values, axis=0)), axis=0)

    polar_coordinates_last = ca.calculate_polar_coordinates_from_trajectory(trajectory, link_lengths)
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


    x_new_data, y_new_data, coefficients = tr.cubic_spline_interpolation(x_data, y_data, new_x_values = np.arange(0, 2.1, 0.1))

    # 计算 F_c，表示运动轨迹的长度
    #F_c = np.sum(np.linalg.norm(np.diff(theta_values, axis=0), axis=1))
    #≈ = pa.cubic_spline_length((a, b, c, d), x_data1[0], x_data1[-1])
    total_length = pa.cubic_spline_length(coefficients, x_data, y_data)
    F_c =total_length
    # 计算 t_T，表示总的运行时间
    t_T = np.sum(t_values)
    # 适应度函数，可根据具体问题进行调整
    total_fitness = 1 / (beta[0] * F_c + beta[1] * t_T + beta[2] * np.sum(F_q))
    return total_fitness


# 交叉操作
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)

    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return child1, child2


# 变异操作
def mutate(individual):
    mutation_point = np.random.randint(0, len(individual))
    individual[mutation_point] += np.random.uniform(-0.1, 0.1)
    return individual

# 遗传算法主循环
# 初始化种群
def run ():
    population = np.array([encode_individual(trajectory(n, b, theta_lower=105/180*np.pi, theta_upper=112/180*np.pi), n) for _ in range(population_size)])

    for generation in range(max_generations):
        # 计算适应度
        fitness_values = np.array([fitness_function(individual) for individual in population])

        # 选择父代
        fitness_sum = np.sum(fitness_values)
        if fitness_sum > 0:
            probabilities = fitness_values / fitness_sum
            probabilities = np.clip(probabilities, 0, 1)  # Clip probabilities to be within [0, 1]
            probabilities[-1] = 1.0 - np.sum(probabilities[:-1])  # Set the last probability to the remaining difference
        else:
            probabilities = np.ones(population_size) / population_size

        # Ensure probabilities are non-negative
        probabilities = np.maximum(probabilities, 0)

        # 重新标准化确保概率之和为1
        probabilities /= np.sum(probabilities)

        # 使用np.random.choice选择父代
        selected_indices = np.random.choice(population_size, size=population_size, p=probabilities)
        selected_population = population[selected_indices]

        # 交叉操作
        crossover_mask = np.random.rand(population_size) < crossover_probability
        children = selected_population.copy()

        # Perform crossover only for selected individuals
        for i in range(0, population_size - 1, 2):
            if crossover_mask[i]:
                children[i], children[i + 1] = crossover(selected_population[i], selected_population[i + 1])

        # 变异操作
        mutation_mask = np.random.rand(population_size) < mutation_probability
        mutation_indices = np.where(mutation_mask)[0]
        for i in mutation_indices:
            children[i] = mutate(children[i])

        # 更新种群
        population = children

        # 输出每代的最优解
        best_individual = population[np.argmax(fitness_values)]
        best_individual_fitness = fitness_function(best_individual)
        print(f"Generation {generation + 1}, Best Fitness: {fitness_function(best_individual)}")
    return best_individual, best_individual_fitness
# 输出最终结果
'''
if __name__ == "__main__":
    best_individual, best_individual_fitness =  run ()
    print("Final Result:")
    print("Best Individual:", best_individual)
    print("Best Fitness:", fitness_function(best_individual))
    
    trajectory_new = decode_individual(best_individual, n)
    polar_coordinates_last = ca.calculate_polar_coordinates_from_trajectory(trajectory_new, link_lengths)
    # 转换为笛卡尔坐标
    cartesian_coordinates_last = [ca.polar_to_cartesian(r, theta) for r, theta in polar_coordinates_last]

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
    print("Best Fitness:", fitness_function(best_individual))
    print("x",x_data)
    print("y",y_data)
    a, b, c, d = tr.cubic_spline_coefficients(x_data, y_data)
    interpolated_curve1 = tr.generate_interpolated_curve(x_data, a, b, c, d)
    # Plot the original data points and the interpolated curve
    plt.scatter(x_data, y_data, label='Data Points')
    plt.plot(interpolated_curve1[:, 0], interpolated_curve1[:, 1], label='Cubic Spline Interpolation', color='red')
    plt.legend()
    plt.show()


    #构建轨迹点
    new_cartesian_coordinates = np.zeros(4)
    new_cartesian_coordinates[0] = ik.forward_kinematics(link_lengths, joint_angles)
    new_cartesian_coordinates[1:] = cartesian_coordinates_last
    #new_cartesian_coordinates[4] = target_position

    print(new_cartesian_coordinates)
'''






