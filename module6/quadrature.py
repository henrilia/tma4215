import numpy as np
from typing import Callable
from scipy.integrate import quad


def lagrange_basis(t: np.ndarray, n: int, i: int):
    y = 1
    for k in range(n):
        if k == i:
            continue

        y *= (t - k) / (i - k)
    return y


def midpoint(a: float, b: float, f: Callable) -> float:
    return (b - a) * f((a + b) / 2)


def trapezoidal(a: float, b: float, f: Callable) -> float:
    return (b - a) / 2 * (f(a) + f(b))


def simpson(a: float, b: float, f: Callable) -> float:
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))


def newton_coates(a: float, b: float, n: int, f: Callable, closed=True) -> float:
    x = np.linspace(a, b, n)
    w = np.zeros_like(x)
    if closed:
        lower = 0
        upper = n
    else:
        lower = -1
        upper = n + 1
    for i in range(len(w)):
        l = lambda t: lagrange_basis(t, n, i)
        w[i] = quad(l, lower, upper)[0]

    return (b - a) / n * sum(w * f(x))


if __name__ == "__main__":

    def func(x):
        return np.sin(x) ** 2

    a, b = (-1, 1)

    print(newton_coates(a, b, 1, func))
    print(newton_coates(a, b, 2, func))
    print(newton_coates(a, b, 4, func))
    print(newton_coates(a, b, 5, func))
    print(newton_coates(a, b, 20, func))
    print(trapezoidal(a, b, func))
