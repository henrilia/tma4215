import numpy as np


def T(x, n=0):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return 2 * x * T(x, n - 1) - T(x, n - 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1)
    N = int(input("N = "))
    for n in range(N + 1):
        Tn = T(x, n)
        plt.plot(x, Tn, label=f"n = {n}")

    plt.grid()
    plt.show()
