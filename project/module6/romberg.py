import numpy as np
from typing import Callable
from project.utils import timeit
from project.module6.quadrature import composite_trapezoidal


@timeit
def romberg(a: float, b: float, n: int, f: Callable):
    """
    romberg function
    """
    A = np.zeros((n))
    for m in range(n):
        A[m] = composite_trapezoidal(a, b, f, m, silent=True)

    for q in range(0, n - 1):
        A = (4 ** (q + 1) * A[1:] - A[:-1]) / (4 ** (q + 1) - 1)

    return A[0]


if __name__ == "__main__":

    def func(x):
        return 4 / (1 + x ** 2)

    a, b = (0, 1)
    print(romberg(a, b, n=2, f=func))
    print(romberg(a, b, n=3, f=func))
    print(romberg(a, b, n=4, f=func))
    print(romberg(a, b, n=5, f=func))
    print(romberg(a, b, n=6, f=func))
    print(romberg(a, b, n=28, f=func))
    print(composite_trapezoidal(a, b, func, 28))
