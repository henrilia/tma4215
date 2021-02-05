import numpy as np
from copy import deepcopy
from typing import Callable, List
from project.module2.factorization import solve


def _copy(func):
    def wrapper(*args, **kwargs) -> Callable:
        d_args: List[np.ndarray] = []
        for arg in enumerate(args):
            d_args.append(deepcopy(arg[1]))
        for kwarg in kwargs:
            kwargs[kwarg] = deepcopy(kwargs[kwarg])
        return func(*d_args, **kwargs)

    return wrapper


@_copy
def tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:

    n = len(b)
    x = np.zeros(n)

    # elimination:

    for k in range(1, n):
        q = a[k] / b[k - 1]
        b[k] = b[k] - c[k - 1] * q
        d[k] = d[k] - d[k - 1] * q

    # backsubstitution:

    q = d[n - 1] / b[n - 1]
    x[n - 1] = q

    for k in range(n - 2, -1, -1):
        q = (d[k] - c[k] * q) / b[k]
        x[k] = q

    return x


if __name__ == "__main__":
    a = np.array([-1.0, 1, 0.1])
    b = np.array([4, 5, 10.1])
    c = np.array([1, 0.1, 0])
    d = np.array([2, 2, 2])

    x = tdma(a=a, b=b, c=c, d=d)
    A = np.diag(b)
    A += np.diag(a[1:], k=-1)
    A += np.diag(c[:-1], k=1)
    print(A, x, A @ x - d, sep="\t\n")
    _x = solve(A, d)
    print(A, _x, A @ _x - d, sep="\t\n")
