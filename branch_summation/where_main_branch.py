import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def _step_algorithm(f, r_min, r_max, freq):
    phi_arr = np.linspace(-np.pi, np.pi, freq)
    r_opt = np.array(tuple(sc.optimize.root_scalar(
        lambda r: f(r * np.exp(1j * phi)), x0=r_min, x1=r_max).root for phi in phi_arr))
    return r_opt, phi_arr


def plot_borders(k, freq):
    r_approx = 2 * np.pi * np.abs(k) + np.pi / 2

    # теперь надо найти для каждого фи такую эр, чтоб была граница между той областью, где брать оптимально k и k+1.
    # для этого сначала напишем re_f_in_z_k, чтоб по Z и k определять значение f
    def re_f_in_z_k(Z, k):
        z_k = 1j * sc.special.lambertw(Z, k=k)
        return np.real(z_k ** 2 / 2j + z_k)

    r_arr, phi_arr = _step_algorithm(lambda z: re_f_in_z_k(z, k) - re_f_in_z_k(z, k + 1), r_approx, r_approx + 2, freq)
    plot_polar(r_arr, phi_arr, c='green')


def _best_k_slow(Z, k_sign):
    k_mean_abs, k_range = np.int_(np.abs(Z) / (2 * np.pi)), 5
    k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range) * k_sign
    z_k = np.array([1j * sc.special.lambertw(Z, k=k) for k in k_arr])
    f_in_z_k = z_k ** 2 / 2j + z_k
    k_max = k_arr[np.argmax(np.real(f_in_z_k * (-k_sign)))]
    return k_max


def plot_best_k(x_step_params, y_step_params, k_sign, alpha: int | float =1):
    """x_step_params is (x_min, x_max, n_dots)"""
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
              cmap='cool', shading='nearest', alpha=alpha)


def plot_sign_f(f, x_step_params, y_step_params):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = np.sign(f(X, Y))
    fig, ax = plt.subplots(1, 1)
    ax.pcolor(X, Y, Z,
              cmap='inferno', shading='nearest')


def plot_polar(r_arr, phi_arr, c=None):
    # plt.polar - отстой
    plt.plot(r_arr * np.cos(phi_arr), r_arr * np.sin(phi_arr), c=c)


def plot_where_biggest_f(r_max, k_sign):
    r_arr = np.linspace(0, r_max, 700)
    phi_arr = k_sign * (r_arr + np.pi / 2)
    plot_polar(r_arr, phi_arr)


def plot_asymptotic_borders(k, freq):
    phi_arr = np.linspace(-np.pi, np.pi, freq)
    r_arr = 2 * np.pi * np.abs(k) + (phi_arr + np.pi / 2) * np.sign(k) + np.pi
    plot_polar(r_arr, phi_arr, c='red')


def plot_difference(x_step_params, y_step_params, k_sign, alpha: int | float =1):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = X + 1j*Y
    diff = np.zeros((x_step_params[2], y_step_params[2]))
    for i in range(x_step_params[2]):
        for j in range(y_step_params[2]):
            # if np.abs(Z[i, j]) < 2:
            #     diff[i, j] = -10
            #     continue
            k_mean_abs, k_range = np.int_((np.abs(Z[i, j])+np.abs(np.angle(Z[i, j])-np.pi/2*k_sign)) / (2 * np.pi)), 3
            k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range + 1) * k_sign
            z_k = np.array([1j * sc.special.lambertw(Z[i, j], k=k) for k in k_arr])
            f_in_z_k = z_k ** 2 / 2j + z_k
            vals = np.real(f_in_z_k * (-k_sign))

            k_max_index = np.argmax(vals)
            # k_max_index = list(k_arr).index(k_mean_abs * k_sign)
            diff[i, j] = np.max(tuple(vals[i] - vals[k_max_index] for i in range(len(vals)) if abs(i - k_max_index) > 1))
    ax = plt.gca()
    ax.pcolor(X, Y, diff,
              cmap='inferno', shading='nearest', alpha=alpha)
    print(np.max(diff), np.argmax(diff))




# next funcs are just for main (plotting in main simply 'mplot')

def mplot_borders_near_minus_axes(Z_range, freq):
    for k_abs in range(int(Z_range / 2 / np.pi)):
        plot_borders(k_abs + 0.0000001, 100)
    plot_best_k((-30, 1, freq), (-0.01, 0.01, 20), -1)


def mplot_line_of_maximums(Z_range, freq):
    plot_best_k((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), -1)
    plot_where_biggest_f(Z_range, -1)


def mplot_asymptotic_border(Z_range, freq):
    plot_best_k((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), -1)
    for i in range(int(Z_range / 2 / np.pi) + 1):
        plot_asymptotic_borders(i - 0.00001, 100)

def mplot_difference_with_borders(Z_range, freq):
    plot_best_k((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), -1, alpha=0.5)
    print('finished plotting best k')
    plot_difference((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), -1, alpha=0.5)


def mplot_annotations_for_k_bar():
    # Z_range = 10
    axes = plt.gca()
    for k, (x, y) in enumerate(((0,0),(-1.8, -2.54), (-6, -6.5), (-10, -10))):
        axes.annotate(r'$\bar{k}='+str(k)+r'$', xy=(x, y), xytext=(x + 0.4, y + 0.2))

def mplot_difference_less_eps(Z_range, freq, eps):
    _diff_less_eps(eps, (-Z_range, Z_range, freq), (-Z_range, Z_range, freq), 1, alpha=1)


def _diff_less_eps(eps, x_step_params, y_step_params, k_sign, alpha: int | float = 1):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = X + 1j*Y
    diff = np.zeros((x_step_params[2], y_step_params[2]))
    for i in range(x_step_params[2]):
        for j in range(y_step_params[2]):
            # if np.abs(Z[i, j]) < 2:
            #     diff[i, j] = -10
            #     continue
            k_mean_abs, k_range = np.int_((np.abs(Z[i, j])+np.abs(np.angle(Z[i, j])-np.pi/2*k_sign)) / (2 * np.pi)), 3
            k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range + 1) * k_sign
            z_k = np.array([1j * sc.special.lambertw(Z[i, j], k=k) for k in k_arr])
            f_in_z_k = z_k ** 2 / 2j + z_k
            vals = np.real(f_in_z_k * (-k_sign))

            if np.min(np.abs(vals[1:] - vals[:-1])) < eps:
                diff[i, j] = -1
            else:
                diff[i, j] = _best_k_slow(X[i, j] + 1j * Y[i, j], k_sign)

    ax = plt.gca()
    ax.pcolor(X, Y, diff,
              cmap='cool', shading='nearest', alpha=alpha)
    print(np.max(diff), np.argmax(diff))

if __name__ == '__main__':
    Z_range, freq = 11, 600
    # plot_difference((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), 1, alpha=1)
    # plot_where_biggest_f(Z_range, -1)
    plot_best_k((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), 1, alpha=1)

    mplot_difference_less_eps(Z_range, freq, 0.02)
    mplot_annotations_for_k_bar()
    plt.show()
