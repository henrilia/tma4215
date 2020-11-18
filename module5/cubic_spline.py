import numpy as np


class CubicSpline:
    def __init__(self, t: np.ndarray, y: np.ndarray):
        if len(set(t)) != len(t):
            raise Exception("Cannot have multiple entries of same x value")
        self.t = t
        self.y = y

        n = len(t) - 1

        self.h = np.zeros(n)
        self.b = np.zeros(n)
        for i in range(n):
            self.h[i] = t[i + 1] - t[i]
            self.b[i] = 6 * (y[i + 1] - y[i]) / self.h[i]

        self.u = np.zeros(n)
        self.v = np.zeros(n)
        self.u[1] = 2 * (self.h[0] + self.h[1])
        self.v[1] = self.b[1] - self.b[0]

        for i in range(2, n):
            self.u[i] = (
                2 * (self.h[i] + self.h[i - 1]) - self.h[i - 1] ** 2 / self.u[i - 1]
            )
            self.v[i] = (
                self.b[i]
                - self.b[i - 1]
                - self.h[i - 1] * self.v[i - 1] / self.u[i - 1]
            )

        self.z = np.zeros(n + 1)
        for i in range(n - 1, 0, -1):
            self.z[i] = (self.v[i] - self.h[i] * self.z[i + 1]) / self.u[i]

    def f(self, x: float):
        if x == self.t[-1]:
            return self.y[-1]

        i = None
        for j in range(len(self.t) - 1):
            if x >= self.t[j] and x < self.t[j + 1]:
                i = j
                break
        else:
            raise Exception("x not in interval of spline")

        return (
            self.z[i] / (6 * self.h[i]) * (self.t[i + 1] - x) ** 3
            + self.z[i + 1] / (6 * self.h[i]) * (x - self.t[i]) ** 3
            + (self.y[i + 1] / self.h[i] - self.z[i + 1] * self.h[i] / 6)
            * (x - self.t[i])
            + (self.y[i] / self.h[i] - self.z[i] * self.h[i] / 6) * (self.t[i + 1] - x)
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def func(x):
        return np.sin(
            x
        )  # 0.001 * x ** 5 - 0.05 * x ** 4 - 10 * x ** 2 + x - 10 + np.sin(x) ** 3

    x0 = np.arange(0, 11, 1)
    y0 = func(x0)

    cs = CubicSpline(x0, y0)

    x = np.arange(0, 10, 0.001)
    y = np.zeros_like(x)

    for i in range(len(y)):
        y[i] = cs.f(x[i])

    plt.plot(x, func(x))
    plt.plot(x, y)
    plt.legend(["Real function", "Spline"])
    plt.title("Cubic spline")
    plt.grid()
    plt.show()