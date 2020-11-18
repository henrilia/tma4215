import numpy as np


def _poly_newton_coefficient(x: np.ndarray, y: np.ndarray):
    m = len(x)

    for k in range(1, m):
        y[k:m] = (y[k:m] - y[k - 1]) / (x[k:m] - x[k - 1])

    return y


def newton_interpolation(x: np.ndarray, x0: np.ndarray, y0: np.ndarray):
    if len(set(x0)) != len(x0):
        raise Exception("Cannot have multiple entries of same x value")

    a = _poly_newton_coefficient(x0, y0)
    n = len(x0) - 1
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (x - x0[n - k]) * p

    return p


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def func(x):
        return x ** 5 - 5 * x ** 4 - 10 * x ** 2 + x - 10

    x0 = np.arange(0, 10, 2)
    y0 = x0 ** 5 - 5 * x0 ** 4 - 10 * x0 ** 2 + x0 - 10

    x = np.arange(0, 10, 0.1)

    y = newton_interpolation(x, x0, y0)

    plt.plot(x, func(x))
    plt.plot(x, y)
    plt.legend(["Real function", "Newton"])
    plt.title("Newton interpolation")
    plt.grid()
    plt.show()
