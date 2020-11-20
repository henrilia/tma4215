import numpy as np
from typing import Callable

from quadrature import composite_trapezoidal


def romberg(a: float, b: float, n: int, f: Callable):
    A = np.zeros((n))
    for m in range(n):
        A[m] = composite_trapezoidal(a, b, f, m)

    for q in range(0, n - 1):
        A = (4 ** (q + 1) * A[1:] - A[:-1]) / (4 ** (q + 1) - 1)

    return A[0]


if __name__ == "__main__":

    def func(x):
        return 4 / (1 + x ** 2)

    a, b = (0, 1)

    print(romberg(a, b, 2, func))
    print(romberg(a, b, 3, func))
    print(romberg(a, b, 4, func))
    print(romberg(a, b, 5, func))
    print(romberg(a, b, 6, func))
    print(romberg(a, b, 28, func))
    print(composite_trapezoidal(a, b, func, 28))
