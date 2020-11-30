import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def solver(y0: np.ndarray, N: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the solved pendulum problem using both Explicit Euler method (y_ee)
    and Symplectic Euler method (y_se).
    """
    h = T / N
    y_ee = np.zeros((2, N))
    y_se = np.zeros((2, N))
    y_ee[:, 0] = y0
    y_se[:, 0] = y0

    for n in range(1, N):
        q_ee = y_ee[0, n - 1] + h * y_ee[1, n - 1]
        p_ee = y_ee[1, n - 1] - h * np.sin(y_ee[0, n - 1])
        y_ee[:, n] = np.array([q_ee, p_ee])

        q_se = y_se[0, n - 1] + h * y_se[1, n - 1]
        p_se = y_se[1, n - 1] - h * np.sin(q_se)
        y_se[:, n] = np.array([q_se, p_se])

    return y_ee, y_se


if __name__ == "__main__":
    N = 1000
    T = 30
    q0 = 1
    p0 = 0

    y0 = np.array([q0, p0])
    y_ee, y_se = solver(y0, N, T)

    plt.plot(y_ee[0, :], y_ee[1, :])
    plt.plot(y_se[0, :], y_se[1, :])
    plt.plot(y0[0], y0[1], "o")
    plt.xlabel("q")
    plt.ylabel("p")
    plt.title("Phase plot of non-linear pendulum problem")
    plt.legend(
        ["Explicit Euler method", "Symplectic Euler method"],
        loc="upper left",
    )
    plt.grid()
    plt.show()