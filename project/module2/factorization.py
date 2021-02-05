import numpy as np
import scipy.linalg as lg
from typing import Tuple
from copy import deepcopy
from typing import Callable


def _is_square(M: np.ndarray) -> bool:
    if M.shape[0] == M.shape[1]:
        return True
    else:
        return False


def is_valid(func):
    def wrapper(M: np.ndarray, *args, **kwargs) -> Callable:
        if not _is_square(M):
            raise Exception("Only square matrices are allowed")

        if lg.det(M) == 0:
            raise Exception("Matrices can not be singular")

        return func(M, *args, **kwargs)

    return wrapper


@is_valid
def forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    L = deepcopy(L)
    b = deepcopy(b)

    n = L.shape[0]
    x = np.zeros_like(b, dtype=np.float)
    x[0] = b[0] / L[0, 0]

    for i in range(1, n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    return x


@is_valid
def backward_sub(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    U = deepcopy(U)
    b = deepcopy(b)

    n = U.shape[0]

    x = np.zeros_like(b, dtype=np.float)
    x[-1] = b[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - U[i, i:] @ x[i:]) / U[i, i]

    return x


@is_valid
def LUP(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    A = deepcopy(A)
    P = np.arange(n)

    for k in range(n):
        l = np.argmax(abs(A[k:, k])) + k
        P[k], P[l] = P[l], P[k]
        A[P[k + 1 :], k] = A[P[k + 1 :], k] / A[P[k], k]
        A[P[k + 1 :], k + 1 :] = A[P[k + 1 :], k + 1 :] - np.outer(
            A[P[k + 1 :], k], A[P[k], k + 1 :]
        )

    return A, P


@is_valid
def cholesky(A: np.ndarray) -> np.ndarray:
    A = deepcopy(A)
    n = A.shape[0]

    for k in range(n - 1):
        if A[k, k] <= 0:
            raise Exception("Null or negative pivot element")

        A[k, k] = np.sqrt(A[k, k])
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]

        for j in range(k + 1, n):
            A[j:, j] = A[j:, j] - A[j:, k] * A[j, k]

    A[-1, -1] = np.sqrt(A[-1, -1])
    return np.tril(A)


def solve(A, b):
    L = cholesky(A)

    y = forward_sub(L, b)
    x = backward_sub(L.T, y)
    return x


if __name__ == "__main__":
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    b = np.array([1, 2, 3])

    L = cholesky(A)

    y = forward_sub(L, b)
    x = backward_sub(L.T, y)

    print(f"A = \n{A}\nCholesky factorization:\nL = \n{L}")
    print(f"b = {b}\nx = {x}")
