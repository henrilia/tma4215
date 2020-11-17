import numpy as np
from typing import Callable


def lagrange(x: np.ndarray, x0: np.ndarray, y0: np.ndarray):
    if len(set(x0)) != len(x0):
        raise Exception("Cannot have multiple entries of same x value")

    k = len(x0)

    L = np.zeros_like(x)
    for j in range(k):
        l = np.ones_like(x)
        for m in range(k):
            if j == m:
                continue

            l *= (x - x0[m]) / (x0[j] - x0[m])

        L += y0[j] * l

    return L


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def func(x):
        return x ** 5 - 5 * x ** 4 - 10 * x ** 2 + x - 10

    x0 = np.arange(0, 10, 2)
    y0 = x0 ** 5 - 5 * x0 ** 4 - 10 * x0 ** 2 + x0 - 10

    x = np.arange(0, 10, 0.1)

    y = lagrange(x, x0, y0)

    plt.plot(x, func(x))
    plt.plot(x, y)
    plt.legend(["Real function", "Langrange"])
    plt.title("Langrange interpolation")
    plt.grid()
    plt.show()
