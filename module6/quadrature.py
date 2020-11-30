import numpy as np
from typing import Callable
from scipy.integrate import quad


VectorFunc = Callable[[np.ndarray], np.ndarray]


def lagrange_basis(t: np.ndarray, n: int, i: int) -> float:
    y = 1
    for k in range(n):
        if k == i:
            continue

        y *= (t - k) / (i - k)
    return y


def midpoint(a: float, b: float, f: Callable[[float], float]) -> float:
    return (b - a) * f((a + b) / 2)


def trapezoidal(a: float, b: float, f: Callable[[float], float]) -> float:
    return (b - a) / 2 * (f(a) + f(b))


def simpson(a: float, b: float, f: Callable[[float], float]) -> float:
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))


def newton_cotes(
    a: float, b: float, n: int, f: Callable[[float], float], closed=True
) -> float:
    x = np.linspace(a, b, n)
    w = np.zeros_like(x)
    if closed:
        lower = 0
        upper = n
        h = (b - a) / n
    else:
        lower = -1
        upper = n + 1
        h = (b - a) / (n + 2)
    for i in range(len(w)):
        l = lambda t: lagrange_basis(t, n, i)
        w[i] = quad(l, lower, upper)[0]

    return (b - a) / n * sum(w * f(x))


def composite_trapezoidal(a: float, b: float, f: VectorFunc, m: int) -> float:
    h = (b - a) / 2 ** m
    arr = np.arange(1, 2 ** m - 1)
    T = np.sum(f(a + arr * h))

    T += 1 / 2 * f(a) + 1 / 2 * f(b)
    T *= h
    return T


if __name__ == "__main__":

    def func(x):
        return np.sin(x) ** 2

    a, b = (-1, 1)

    print(newton_cotes(a, b, 1, func))
    print(newton_cotes(a, b, 2, func))
    print(newton_cotes(a, b, 4, func))
    print(newton_cotes(a, b, 5, func))
    print(newton_cotes(a, b, 20, func))
    print(trapezoidal(a, b, func))
    print(composite_trapezoidal(a, b, func, 10))
