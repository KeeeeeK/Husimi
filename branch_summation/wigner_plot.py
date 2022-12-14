import numpy as np
import matplotlib.pyplot as plt
from wigner_calculus import nb_wigner as wigner

def test_plot_wigner(x_step_params, y_step_params, alpha, gamma):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    W = wigner(np.abs(alpha), np.angle(alpha), np.abs(X+1j*Y), np.angle(X+1j*Y), gamma, 4)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, W, cmap='inferno')



if __name__ == '__main__':
    xy_range = 5
    x_range, y_range, freq = xy_range, xy_range, 400
    test_plot_wigner((-x_range, x_range, freq), (-y_range, y_range, freq), 4+1j, 1)
    plt.show()
