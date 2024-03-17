import numpy as np
import matplotlib.pyplot as plt


def solve_tridiagonal_system(A, b):
    """
    Solve a tridiagonal system of equations Ax = b using Gaussian elimination.
    A is a tridiagonal matrix represented by a 2D numpy array.
    b is a 1D numpy array.
    """
    n = len(b)

    # Forward elimination
    for i in range(1, n):
        factor = A[i][i - 1] / A[i - 1][i - 1]
        A[i] -= factor * A[i - 1]
        b[i] -= factor * b[i - 1]

    # Backward substitution
    x = np.zeros(n)
    x[-1] = b[-1] / A[-1][-1]

    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - A[i][i + 1] * x[i + 1]) / A[i][i]

    return x


def cubic_spline_coefficients(x, y):
    """
    Compute the coefficients (a, b, c, d) for the cubic spline interpolation.
    """
    n = len(x)
    h = np.diff(x)

    # Build the tridiagonal matrix
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1:i + 2] = [h[i - 1], 2 * (h[i - 1] + h[i]), h[i]]

    # Build the right-hand side vector
    delta_y = np.diff(y)
    b = np.zeros(n)
    b[1:-1] = 3 * (delta_y[1:] / h[1:] - delta_y[:-1] / h[:-1])

    # Solve the tridiagonal system
    c = solve_tridiagonal_system(A, b)

    # Calculate the remaining coefficients
    a = y[:-1]
    b = (delta_y / h) - (h / 3) * (2 * c[:-1] + c[1:])
    d = (c[1:] - c[:-1]) / (3 * h)

    return a, b, c[1:], d



def cubic_spline_interpolation(x, a, b, c, d, x_i):
    """
    Compute the cubic spline interpolation at a given point x_i.
    """
    delta_x = x_i - x
    return a + b * delta_x + c * delta_x**2 + d * delta_x**3

def generate_interpolated_curve(x, a, b, c, d):
    """
    Generate points on the entire interpolated curve.
    """
    interpolated_curve = []
    for i in range(len(x) - 1):
        x_interval = np.linspace(x[i], x[i+1], 100)
        y_interval = cubic_spline_interpolation(x[i], a[i], b[i], c[i], d[i], x_interval)
        interpolated_curve.extend(list(zip(x_interval, y_interval)))
    return np.array(interpolated_curve)

# Example usage:
x_data = np.array([0.3, 0.8, 1.3, 2])
y_data = np.array([2, 1, 4, 1])

a, b, c, d = cubic_spline_coefficients(x_data, y_data)

# Generate interpolated curve
interpolated_curve = generate_interpolated_curve(x_data, a, b, c, d)

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

# Plot the original data points and the interpolated curve
plt.scatter(x_data, y_data, label='Data Points')
plt.plot(interpolated_curve[:, 0], interpolated_curve[:, 1], label='Cubic Spline Interpolation', color='red')
plt.legend()
plt.show()







