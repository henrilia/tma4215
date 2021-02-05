import numpy as np
import scipy.linalg as lg
from typing import Tuple
from project.utils import timeit, VectorFunction


def jacobian(F: np.ndarray, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros_like(x)
        dx[j] = h
        J[:, j] = (F(x + dx) - F(x - dx)) / (2 * h)

    return J


@timeit
def approx_newton(
    x: np.ndarray, F: VectorFunction, damp: float = 1.0, tol: float = 1e-8, p: int = 1
) -> Tuple[np.ndarray, int]:

    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e5:
        if iter % p == 0:
            J = jacobian(F, x)

        x += damp * lg.solve(J, -F(x))
        iter += 1

    return x, iter


@timeit
def quasi_newton(
    x: np.ndarray,
    F: VectorFunction,
    J: VectorFunction,
    damp: float = 1.0,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, int]:

    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e4:
        x += damp * lg.solve(J(x), -F(x))
        iter += 1

    return x, iter


@timeit
def broyden(
    x: np.ndarray,
    F: VectorFunction,
    B: np.ndarray,
    damp: float = 1.0,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, int]:

    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e4:
        dx = damp * lg.solve(B, -F(x))
        s = damp * dx
        y = F(x + s) - F(x)
        x += s
        B += np.outer((y - B @ s), s) / (np.inner(s, s))
        iter += 1

    return x, iter


@timeit
def fixed_point(
    x: np.ndarray,
    F: VectorFunction,
    G: VectorFunction,
    damp: float = 1.0,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, int]:

    iter = 0
    while lg.norm(F(x)) > tol and iter < 1e4:
        x = G(x)
        iter += 1

    return x, iter


if __name__ == "__main__":

    def F(x):
        return np.array([x[0] ** 2 * x[1], 5 * x[0] + np.sin(x[1])])

    def J_F(x):
        return np.array([[2 * x[0] * x[1], x[0] ** 2], [5, np.cos(x[1])]])

    x0 = np.array([0.4, 0.1])
    x = quasi_newton(x0, F, J_F, damp=1.2)
    print(*x)

    x0 = np.array([0.4, 0.1])
    x = quasi_newton(x0, F, J_F, damp=1.0)
    print(*x)

    x0 = np.array([0.4, 0.1])
    x = approx_newton(x0, F, damp=0.9, p=1)
    print(*x)

    x0 = np.array([0.4, 0.1])
    x = quasi_newton(x0, F, J_F, damp=0.9)
    print(*x)

    x0 = np.array([0.1, 0.1])
    x = broyden(x0, F, J_F(x0), damp=1.0)
    print(*x)
