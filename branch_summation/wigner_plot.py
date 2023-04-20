import numpy as np
import matplotlib.pyplot as plt
from wigner_calculus import nb_wigner as wigner
from wigner_calculus import fast_wigner

def test_plot_wigner(x_step_params, y_step_params, alpha, gamma, n_sigma):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    W = fast_wigner(np.abs(alpha), np.angle(alpha), np.abs(X+1j*Y), np.angle(X+1j*Y), gamma, n_sigma)
    # W0 = 1/np.pi*np.exp(-2*np.abs(alpha - (X+1j*Y))**2)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, W, cmap='inferno')
    # ax.plot_surface(Y, X, W0, cmap='inferno')


def n_sigma_dependence():
    x = np.array((0, 0.1, 0.5, 0.6, 1, 1.5, 1.7, 1.9, 2, 2.3))
    y = wigner(10, 1, 10, 1, 0, 10 ** x)
    plt.plot(x, y)

def plot_fast_wigner(alpha_abs, alpha_arg, beta_arg, gamma, n_sigma):
    x = np.linspace(alpha_abs-1, alpha_abs+1, 400)
    # x = np.linspace(99.4, 99.7, 100)
    y = fast_wigner(alpha_abs, alpha_arg, x, beta_arg, gamma, n_sigma)

    plt.plot(x-alpha_abs, y)

if __name__ == '__main__':
    # alpha = 10
    #
    # xy_range = alpha + np.sqrt(np.abs(alpha))
    # x_range, y_range, freq = xy_range, xy_range, 20
    # test_plot_wigner((-x_range, x_range, freq), (-y_range, y_range, freq), alpha, 0, 100)
    #
    # # n_sigma_dependence()
    # plt.show()
    gamma = 10 ** -6
    alpha = 2/np.sqrt(gamma)
    plot_fast_wigner(alpha, 0, 2*alpha**2*gamma, gamma, 5)
    # plt.gcf().set_size_inches(10, 5)
    plt.show()
    # plt.savefig(fname='1')