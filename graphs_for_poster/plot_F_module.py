import numpy as np
import matplotlib.pyplot as plt
from only_steepest_descent.plot import plot_beauty
from numerical_summation.nb_sum_fn import sum_fn
from decimal import Decimal

n_sigma = 5

def fix_axes():
    x_min, x_max = -n_sigma, n_sigma
    y_min, y_max = 0, 1.04
    axes = plt.gca()
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)

def set_label():
    axes = plt.gca()
    for set_label, axis, label, label_coords in (
            (axes.set_xlabel, axes.xaxis,
             r'\frac{\arg A + 2 \left|A\right| \Gamma + \Gamma}{2\sqrt{\left|A\right|}\Gamma}', np.array([1.03, -0.03])),
            (axes.set_ylabel, axes.yaxis,
             r'\frac{\left|F(A, e^{i\Gamma})\right|}{F_{\max}}', np.array([-0.06, 1]))):

        label_prop = dict(rotation=0)
        set_label('$' + label + '$', label_prop)
        axis.set_label_coords(*label_coords)


@plot_beauty
def plot_F():
    Gamma_A_abs_and_G_arr = ((1, 10 ** -6), (1.1, 10 ** -6), (9, 10 ** -5), (1.2, 10 ** -5),
                             (2.5, 10 ** -4), (5, 10 ** -4), (5, 10 ** -3))
    for Gamma_A_abs, G in Gamma_A_abs_and_G_arr:
        A_abs = Gamma_A_abs / abs(G)
        Fn_max = 1/np.sqrt(2 * Gamma_A_abs) * (1 - 1/(4*Gamma_A_abs)**2 + 5/2 / (4*Gamma_A_abs) ** 4)

        sigma = 2 * np.sqrt(A_abs) * abs(G)
        shift = np.mod(2*Gamma_A_abs - G, 2*np.pi)
        phi_shifted = np.linspace(-n_sigma * sigma , n_sigma * sigma, 1000)
        phi = phi_shifted - shift

        Fn_values = sum_fn(A_abs, phi, G, 5)
        legend = f'\\left|A\\Gamma\\right|={float(Gamma_A_abs)},\\; \\Gamma=1E-{round(-np.log10(G))}'
        plt.plot(phi_shifted / sigma, np.abs(Fn_values) / Fn_max, label = '$' + legend + '$')



fix_axes()
set_label()
plot_F()
plt.legend()
plt.savefig('foo.png')