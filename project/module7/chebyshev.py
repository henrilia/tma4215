import numpy as np
from scipy.integrate import quad

from project.utils import VectorFunction


def nodes(n: int) -> np.ndarray:
    x = np.zeros(n)

    for k in range(n):
        x[k] = np.cos((2 * k + 1) / (2 * n) * np.pi)

    return x


def coefficients(n: int) -> np.ndarray:
    return np.ones(n) * np.pi / n


def quadrature(f: VectorFunction, n: int) -> float:
    x = nodes(n)
    w = coefficients(n)
    return np.sum(w * f(x))


def T(x: np.ndarray, n: int = 0, recursive: bool = False) -> np.ndarray:
    if recursive:
        return _T_recursion(x, n)
    else:
        return _T(x, n)


def _T(x: np.ndarray, n: int) -> np.ndarray:
    return np.cos(n * np.arccos(x))


def _T_recursion(x: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return 2 * x * _T_recursion(x, n - 1) - _T_recursion(x, n - 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1)
    N = int(input("N = "))

    plt.figure(1)
    for n in range(N + 1):
        Tn = T(x, n)
        plt.plot(x, Tn, label=f"n = {n}")

    plt.title("Explicit formula")
    plt.grid()
    plt.show()

    plt.figure(2)
    Tn = None
    for n in range(N + 1):
        Tn = T(x, n, recursive=True)
        plt.plot(x, Tn, label=f"n = {n}")

    plt.title("Recursive formula")
    plt.grid()
    plt.show()

    def func(x):
        return 1 / np.sqrt(1 + 5 * x ** 2)  # (np.cos(x) ** 2) / np.sqrt(1 + x ** 2)

    def w_func(x):
        return func(x) / np.sqrt(1 - x ** 2)

    n = N
    plt.figure(3)
    plt.plot(x, Tn)
    x = np.linspace(-1, 1, 100)
    plt.plot(x, func(x))
    x_c = nodes(n)
    plt.vlines(x_c, ymin=np.zeros(n), ymax=func(x_c))
    plt.grid()
    plt.show()

    print(quad(w_func, -1, 1)[0])
    print(quadrature(func, n=7))
