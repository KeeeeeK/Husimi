import numpy as np
import matplotlib.pyplot as plt
from wigner_calculus import nb_wigner as wigner

def test_plot_wigner(x_step_params, y_step_params, alpha, gamma, n_sigma):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    W = wigner(np.abs(alpha), np.angle(alpha), np.abs(X+1j*Y), np.angle(X+1j*Y), gamma, n_sigma)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, W, cmap='inferno')


def n_sigma_dependence():
    x = np.array((0, 0.1, 0.5, 0.6, 1, 1.5, 1.7, 1.9, 2, 2.3))
    y = wigner(10, 1, 10, 1, 0, 10 ** x)
    plt.plot(x, y)


if __name__ == '__main__':
    alpha = 10 ** (2/3)

    xy_range = alpha + np.sqrt(np.abs(alpha))+1
    x_range, y_range, freq = xy_range, xy_range, 20
    test_plot_wigner((-x_range, x_range, freq), (-y_range, y_range, freq), alpha, 0, 100)

    # n_sigma_dependence()
    plt.show()
