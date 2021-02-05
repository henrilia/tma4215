import numpy as np


def L(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    else:
        return (2 * (k - 1) + 1) / k * x * L(x, k - 1) - (k - 1) / (k) * L(x, k - 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1)
    N = int(input("N = "))

    plt.figure(1)
    for k in range(N + 1):
        Lk = L(x, k)
        plt.plot(x, Lk, label=f"k = {k}")

    plt.title("Legendre polynomials")
    plt.grid()
    plt.legend()
    plt.show()
