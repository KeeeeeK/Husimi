import matplotlib.patches as _mp
import matplotlib.pyplot as plt
import numpy as np


def plot_curve(points: np.ndarray[np.ndarray, np.ndarray]):
    plot_beauty()
    plt.plot(*zip(*points))



def plot_beauty():
    axes = plt.gca()
    _grid_lines(axes)
    _arrows(axes)


def _grid_lines(axes):
    axes.grid(axis='both', which='major', linestyle='--', linewidth=1)
    axes.grid(axis='both', which='minor', linestyle='--', linewidth=0.5)
    axes.minorticks_on()


def _fix_axes(axes, zero_in_corner=True):
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    if zero_in_corner is True:
        x_min = min(0, x_min)
        y_min = min(0, y_min)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    return x_min, x_max, y_min, y_max


def _arrows(axes):
    arrowprops = dict(arrowstyle=_mp.ArrowStyle.CurveB(head_length=1), color='black')
    axes.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(-0.03, 0), arrowprops=arrowprops)
    axes.annotate('', xy=(0, 1.06), xycoords='axes fraction', xytext=(0, -0.03), arrowprops=arrowprops)
