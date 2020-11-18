from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt

from cubic_spline import CubicSpline
from lagrange import lagrange


def func(x):
    return erf(x)


t = np.arange(-3, 3 + 0.1, 0.5)

f = func(t)

x = np.arange(-3, 3, 0.02)

y_poly = lagrange(x, t, f)

cs = CubicSpline(t, f)
y_cspline = cs.approximate(x)

plt.figure(1)
plt.plot(x, func(x))
plt.plot(x, y_poly)
plt.plot(x, y_cspline)
plt.title("Approximation")
plt.grid()
plt.legend(["Real solution", "Polynomial interpolation", "Cubic spline interpolation"])
plt.show()

plt.figure(2)
plt.plot(x, func(x) - y_poly)
plt.plot(x, func(x) - y_cspline)
plt.title("Error")
plt.grid()
plt.legend(["Polynomial interpolation", "Cubic spline interpolation"])
plt.show()