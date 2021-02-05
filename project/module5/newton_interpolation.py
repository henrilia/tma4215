import numpy as np

from project.utils import FloatFunction


def divided_difference(x: np.ndarray, f: FloatFunction) -> float:
    if type(x) is not np.ndarray:
        return f(x)
    elif len(x) == 1:
        return f(x)
    elif len(x) == 2:
        return (f(x[1]) - f(x[0])) / (x[1] - x[0])
    else:
        return (divided_difference(x[1:], f) - divided_difference(x[:-1], f)) / (
            x[-1] - x[0]
        )


def divided_difference_y(x: np.ndarray, y: np.ndarray) -> float:
    if type(x) is not np.ndarray:
        return y
    elif len(x) == 1:
        return y
    elif len(x) == 2:
        return (y[1] - y[0]) / (x[1] - x[0])
    else:
        return (
            divided_difference_y(x[1:], y[1:]) - divided_difference_y(x[:-1], y[:-1])
        ) / (x[-1] - x[0])


def newton_interpolation(x: np.ndarray, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    N = np.ones_like(x) * divided_difference_y(x0[0], y0[0])
    for j in range(1, len(x0) + 1):
        a = divided_difference_y(x0[:j], y0[:j])
        n = np.ones_like(x)

        for i in range(j - 1):
            n = n * (x - x0[i])

        N += a * n
    return N


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def func(x):
        return x ** 5 - 5 * x ** 4 - 10 * x ** 2 + x - 10

    x0 = np.arange(0, 10, 2)
    y0 = func(x0)

    x = np.arange(0, 10, 0.1)

    y = newton_interpolation(x, x0, y0)

    plt.plot(x, func(x))
    plt.plot(x, y)
    plt.legend(["Real function", "Newton"])
    plt.title("Newton interpolation")
    plt.grid()
    plt.show()
