import numpy as np
import matplotlib.pyplot as plt

from project.module5.cubic_spline import CubicSpline
from project.module5.lagrange_interpolation import lagrange


def func(x):
    return 1 / (1 + 25 * x ** 2)


t = np.arange(-1.5, 1.5 + 0.1, 0.1)

f = func(t)

x = np.arange(-1, 1, 0.02)

y = func(x)

y_poly = lagrange(x, t, f)

cs = CubicSpline(t, f)
y_cspline = cs.approximate(x)

plt.figure(1)
plt.plot(x, y)
plt.plot(x, y_poly)
plt.plot(x, y_cspline)
plt.title("Approximation")
plt.grid()
plt.legend(["Real solution", "Polynomial interpolation", "Cubic spline interpolation"])
plt.show()

plt.figure(2)
plt.plot(x, y - y_poly)
plt.plot(x, y - y_cspline)
plt.title("Error")
plt.grid()
plt.legend(["Polynomial interpolation", "Cubic spline interpolation"])
plt.show()