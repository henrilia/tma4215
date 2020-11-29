import numpy as np
import scipy.linalg as la


def conjugate_gradient(A_func, b, x0, tol, pre_cond):
    N = len(x0)
    i = 0
    r = b - A_func(x0)
    d = np.array(pre_cond(r))
    delta_new = np.inner(r, d)
    x = x0
    while i < N and tol < la.norm(r):
        q = A_func(d)
        alpha = delta_new / np.inner(d, q)
        x = np.add(x, alpha * d)
        r = np.add(r, -alpha * q)
        s = pre_cond(r)
        delta_old = delta_new
        delta_new = np.inner(r, s)
        beta = delta_new / delta_old
        d = np.add(s, beta * d)
        i += 1

    return x, i, r


def identity(r):
    return r


def pre_cond_jac(v):
    u = np.zeros_like(v)
    for i in range(jac_iter):
        u += 1 / 4 * (v - A_func(u))

    return u


if __name__ == "__main__":

    def A_func(v):
        n2 = len(v)
        n = int(np.sqrt(n2))
        if n ** 2 != n2:
            raise Exception("Must be a square matrix!")
        M = np.zeros((n + 2, n + 2))
        M[1:-1, 1:-1] = np.reshape(v, (n, n))
        AM = 4 * M[1:-1, 1:-1] - M[:-2, 1:-1] - M[2:, 1:-1] - M[1:-1, :-2] - M[1:-1, 2:]
        Av = np.reshape(AM, (n2,))
        return Av

    def b_def(n):
        h = 1 / (n + 1)
        X = np.linspace(h, 1 - h, n)
        Y = X
        B = np.outer(16 * X ** 2 * (1 - X) ** 2, 16 * Y ** 2 * (1 - Y) ** 2)
        b = np.reshape(B, (n ** 2,))
        return B, b

    tol = 10 ** -5

    for n in range(20, 220, 20):
        x0 = np.zeros(n ** 2)
        B, b = b_def(n)

        x, iterNull, _ = conjugate_gradient(A_func, b, x0, tol, identity)
        jac_iter = 2
        x, iterJac2, _ = conjugate_gradient(A_func, b, x0, tol, pre_cond_jac)
        jac_iter = 4
        x, iterJac4, _ = conjugate_gradient(A_func, b, x0, tol, pre_cond_jac)
        print(n, iterNull, iterJac2, iterJac4, sep="\t\t")
