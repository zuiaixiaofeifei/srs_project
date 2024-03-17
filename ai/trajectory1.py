import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def cubic_spline_interpolation(x, y, new_x_values):
# 使用 CubicSpline 进行插值
    cs = CubicSpline(x, y)

# 生成更密集的 x 值，用于绘制光滑曲线
    x_new = np.arange(0, 2.1, 0.1)

# 计算对应的 y 值
    y_new = cs(x_new)

# 获取插值函数的系数值
    coefficients = cs.c
    return x_new, y_new, coefficients
'''
# 创建一些示例数据点
x = np.array([0.3, 0.8, 1.3, 2])
y = np.array([2, 1, 4, 1])
x_new, y_new, coefficients = cubic_spline_interpolation(x, y, new_x_values = np.arange(0, 2.1, 0.1))
# 打印系数值
print("Coefficients for each interval:")
for i, coef in enumerate(coefficients):
    print(f"Interval {i + 1}: {coef}")
a, b, c, d = coefficients
print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)
# 绘制原始数据点和拟合曲线
plt.scatter(x, y, label='Data Points')
plt.plot(x_new, y_new, label='Cubic Spline Fit')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Cubic Spline Interpolation')
plt.show()
'''