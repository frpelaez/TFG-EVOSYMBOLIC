import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Callable

def rk4(f: Callable, t_span: Tuple[float, float], n_points: int, x_0: List[float] | np.ndarray) -> np.ndarray:
    
    t_start, t_end = t_span
    t = np.linspace(t_start, t_end, n_points)
    h = (t_end - t_start) / n_points
    
    x = np.zeros((n_points, len(x_0)))
    x[0] = x_0
    for i in range(n_points - 1):
        k1 = f(t[i], x[i])
        k2 = f(t[i] + .5 * h, x[i] + .5 * h * k1)
        k3 = f(t[i] + .5 * h, x[i] + .5 * h * k2)
        k4 = f(t[i] + h, x[i] + h * k3)
        x[i + 1] = x [i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x

def main() -> None:
    
    alpha = 1
    beta = 1
    gamma = 1
    delta = 1
    f = lambda t, x : np.array([x[0] * (alpha - beta * x[1]),
                                                            -x[1] * (gamma - delta * x[0])])
    
    N = 5_000
    t_span = (0, 30)
    t = np.linspace(t_span[0], t_span[1], N)
    x_0 = np.array([1, 2])
    
    res = rk4(f, t_span, N, x_0)
    
    x1 = res[:, 0]
    x2 = res[:, 1]
    
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # # Plot the 2D trajectory on the first subplot
    # axs[0].plot(x1, x2)
    # axs[0].set_xlabel("x(t)")
    # axs[0].set_ylabel("y(t)")
    # axs[0].set_title("2D Trajectory")
    # axs[0].grid(True)
    
    # # Plot x1 and x2 over time on the second subplot
    # axs[1].plot(t, x1, label="x(t)")
    # axs[1].plot(t, x2, label="y(t)")
    # axs[1].set_xlabel("t")
    # axs[1].set_ylabel("Values")
    # axs[1].set_title("x(t) and y(t) over time")
    # axs[1].legend()
    # axs[1].grid(True)
    
    fig = plt.figure()
    ax1 = fig.add_axes((0.1,0.1,0.8,0.8))
    ax1.set_title("Diagrama de fases")
    ax1.set_xlabel('Población A(t)')
    ax1.set_xlabel('Población B(t)')
    ax1.plot(x1, x2)
    
    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()