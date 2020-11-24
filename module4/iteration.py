import numpy as np
import scipy.linalg as lg
from typing import Callable


def quasi_newton(
    x: np.ndarray, F: Callable, J: Callable, damp: float = 1.0, tol: float = 1e-8
) -> np.ndarray:

    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e4:
        x += damp * lg.solve(J(x), -F(x))
        iter += 1

    return x


def broyden(
    x: np.ndarray, F: Callable, B: np.ndarray, damp: float = 1.0, tol: float = 1e-8
) -> np.ndarray:

    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e4:
        dx = damp * lg.solve(B, -F(x))
        s = damp * dx
        y = F(x + s) - F(x)
        x += s
        B += np.outer((y - B @ s), s) / (np.inner(s, s))
        iter += 1

    return x


def fixed_point(
    x: np.ndarray, F: Callable, G: Callable, damp: float = 1.0, tol: float = 1e-8
) -> np.ndarray:
    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e4:
        x = G(x)
        iter += 1

    return x


if __name__ == "__main__":

    def F(x):
        return np.array([x[0] ** 2 * x[1], 5 * x[0] + np.sin(x[1])])

    def J_F(x):
        return np.array([[2 * x[0] * x[1], x[0] ** 2], [5, np.cos(x[1])]])

    x0 = np.array([0.1, 0.1])
    print(x0)
    x = quasi_newton(x0, F, J_F)
    print(x)

    x0 = np.array([0.1, 0.1])
    print(x0)
    x = broyden(x0, F, J_F(x0))
    print(x)