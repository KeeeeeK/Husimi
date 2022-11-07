import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def _step_algorithm(f, r_min, r_max, freq):
    phi_arr = np.linspace(-np.pi, np.pi, freq)
    r_opt = np.array(tuple(sc.optimize.root_scalar(
        lambda r: f(r*np.exp(1j*phi)), x0=r_min, x1=r_max).root for phi in phi_arr))
    return r_opt, phi_arr



def plot_borders(k, freq):
    r_approx = 2 * np.pi * np.abs(k) + np.pi / 2
    # теперь надо найти для каждого фи такую эр, чтоб была граница между той областью, где брать оптимально k и k+1.
    # для этого сначала напишем re_f_in_z_k, чтоб по Z и k определять значение f
    def re_f_in_z_k(Z, k):
        z_k = 1j*sc.special.lambertw(Z, k=k)
        return np.real(z_k ** 2 / 2j + z_k)
    r_arr, phi_arr = _step_algorithm(lambda z: re_f_in_z_k(z, k) - re_f_in_z_k(z, k+1), r_approx, r_approx+2, freq)
    plot_polar(r_arr, phi_arr, c='green')

def _best_k_slow(Z, k_sign):
    k_mean_abs, k_range = np.int_(np.abs(Z)/(2*np.pi)), 5
    k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range) * k_sign
    z_k = np.array([1j * sc.special.lambertw(Z, k=k) for k in k_arr])
    f_in_z_k = z_k ** 2 / 2j + z_k
    k_max = k_arr[np.argmax(np.real(f_in_z_k * (-k_sign)))]
    return k_max

def plot_best_k(x_step_params, y_step_params, k_sign):
    # print(tuple(it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))))
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    K = np.zeros((x_step_params[2], y_step_params[2]))
    for i in range(x_step_params[2]):
        for j in range(y_step_params[2]):
            # следующая версия правильна, но если комплексно сопрячь, то результат не зависит от k_sign
            # K[i, j] = _best_k(X[i, j] + 1j * Y[i, j], k_sign)
            K[i, j] = _best_k_slow(X[i, j] + 1j * Y[i, j], k_sign)
    # Z = [i+k for i, k in it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))]
    ax = plt.gca()
    ax.pcolor(X, Y, K,
              cmap='inferno', shading='nearest', alpha=1)


def plot_sign_f(f, x_step_params, y_step_params):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = np.sign(f(X, Y))
    fig, ax = plt.subplots(1, 1)
    ax.pcolor(X, Y, Z,
                    cmap='inferno', shading='nearest')

def plot_polar(r_arr, phi_arr, c=None):
    # plt.polar - отстой
    plt.plot(r_arr*np.cos(phi_arr), r_arr*np.sin(phi_arr), c=c)


def plot_where_biggest_f(r_max, k_sign):
    r_arr = np.linspace(0, r_max, 700)
    phi_arr = k_sign * (r_arr + np.pi /2)
    plot_polar(r_arr, phi_arr)


def plot_asymptotic_borders(k, freq):
    phi_arr = np.linspace(-np.pi, np.pi, freq)
    r_arr = 2*np.pi*np.abs(k) + phi_arr * np.sign(k) + np.pi/2
    plot_polar(r_arr, phi_arr, c='red')

if __name__ == '__main__':
    # print(_step_algorithm(lambda z: np.abs(z)-1.2, 1, 2, 10))

    Z_range, freq  = 50, 400
    plot_best_k((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), 1)
    # plot_where_biggest_f(Z_range, -1)
    # for k_abs in range(int(Z_range / 2/np.pi)):
    #     plot_borders(k_abs + 0.0000001, 100)
    # _main_branch_curves(1)
    # plot_best_k((-30, 1, freq), (-0.01, 0.01, 20), -1)
    for i in range(int(Z_range / 2 / np.pi) + 1):
        plot_borders(i, 100)
    plt.show()

    # phi = np.linspace(-np.pi, np.pi)
    # plot_polar(10+phi, phi)
    # plt.show()