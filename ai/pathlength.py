import numpy as np

def cubic_spline_length(coefficients, x_data, y_data):
    total_length = 0.0
    a, b, c, d = coefficients

    for i in range(len(a)):
        x1, x2 = x_data[i], x_data[i + 1]
        y1, y2 = y_data[i], y_data[i + 1]

        # 定义函数 f(x) 和其导数 f'(x)
        def f(x): return a[i] * x**3 + b[i] * x**2 + c[i] * x + d[i]
        def df(x): return 3 * a[i] * x**2 + 2 * b[i] * x + c[i]

        # 使用龙格-库塔法（Runge-Kutta method）进行积分
        n = 1000  # 可以根据需要调整步数
        dx = (x2 - x1) / n
        arc_length = 0.0

        for j in range(n):
            x_curr = x1 + j * dx
            y_curr = f(x_curr)
            y_prime = df(x_curr)

            arc_length += np.sqrt(1 + y_prime**2) * dx

        total_length += arc_length

    return total_length


'''
# 示例用法
a = np.array([1, -1, 0])
b = np.array([0, 2, -1])
c = np.array([0, 0, 1])
d = np.array([0, 1, 0])
x_data = np.array([0, 1, 2, 3])
y_data = np.array([0, 1, 0, -1])

length = cubic_spline_length(a, b, c, d, x_data, y_data)
print("弧长:", length)
'''