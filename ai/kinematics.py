import numpy as np

class Kinematics:
    def __init__(self, link_lengths):
        self.link_lengths = link_lengths

    def forward_kinematics(self, joint_angles):
        """
        正运动学模型
        :param joint_angles: 关节角度，[theta1, theta2, theta3]
        :return: 末端执行器的坐标 (x, y)
        """
        theta1, theta2, theta3 = joint_angles
        L1, L2, L3 = self.link_lengths

        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2) + L3 * np.sin(theta1 + theta2 + theta3)

        return x, y

    def inverse_kinematics(self, target_position):
        """
        逆运动学模型
        :param target_position: 末端执行器的目标坐标 (x, y)
        :return: 关节角度 [theta1, theta2, theta3] 或 None（如果无解）
        """
        x, y = target_position
        L1, L2, L3 = self.link_lengths

        # 计算 theta1
        theta1 = np.arctan2(y, x)  # arctan2 返回的角度范围是 [-pi, pi]

        # 计算余弦定理的分母部分
        D = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)

        # 计算 theta3
        theta3 = np.arctan2(np.sqrt(1 - D ** 2), D)

        # 计算 theta2
        theta2 = np.arctan2(y - L1 * np.sin(theta1), x - L1 * np.cos(theta1)) - np.arctan2(L3 * np.sin(theta3),
                                                                                           L2 + L3 * np.cos(theta3))

        # 返回关节角度
        return theta1, theta2, theta3

    def is_solution_within_joint_limits(self, joint_angles):
        """
        检查逆运动学解是否在关节角度范围内
        :param joint_angles: 关节角度 [theta1, theta2, theta3]
        :return: True 或 False
        """
        # Extract joint angle limits
        theta1_min, theta1_max = -np.pi, np.pi
        theta2_min, theta2_max = -np.pi, np.pi
        theta3_min, theta3_max = -np.pi, np.pi

        # Extract joint angles
        theta1, theta2, theta3 = joint_angles

        # Check if each joint angle is within its limits
        within_limits = (
                theta1_min <= theta1 <= theta1_max and
                theta2_min <= theta2 <= theta2_max and
                theta3_min <= theta3 <= theta3_max
        )

        return within_limits

if __name__ == "__main__":
    link_lengths = [1, 1, 0.5]
    kinematics_example = Kinematics(link_lengths)

    joint_angles = np.radians([45, 45, 45])
    end_effector_pos_example = kinematics_example.forward_kinematics(joint_angles)

    print("Forward Kinematics - Joint Angles:", np.degrees(joint_angles))
    print("Forward Kinematics - End Effector Position:", end_effector_pos_example)

    # 逆运动学：给定末端执行器位置，计算关节角
    target_position = np.array([0.2, 2])  # 你可以修改为你想要到达的坐标
    # Using the end effector position from forward kinematics to compute inverse kinematics
    inverse_kinematics_example = kinematics_example.inverse_kinematics(target_position)

    print("Inverse Kinematics - End Effector Position:", target_position)
    print("Inverse Kinematics - Joint Angles:", np.degrees(inverse_kinematics_example))

