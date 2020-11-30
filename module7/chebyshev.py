import numpy as np


def T(x, n=0, recursive=False):
    if recursive:
        return _T_recursion(x, n)
    else:
        return _T(x, n)


def _T(x, n):
    return np.cos(n * np.arccos(x))


def _T_recursion(x, n):
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
    for n in range(N + 1):
        Tn = T(x, n, recursive=True)
        plt.plot(x, Tn, label=f"n = {n}")

    plt.title("Recursive formula")
    plt.grid()
    plt.show()