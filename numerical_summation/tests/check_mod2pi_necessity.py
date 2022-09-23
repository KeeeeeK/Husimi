# Данный файл посвящён вычислению np.exp(i*phi) при больших phi.
# Насколько велики погрешности? Насколько быстро вычисляется? И можно ли это исправить простым использованием
# np.exp(1j*(np.mod(phi, 2*np.pi)))
#
# Результат: да, предложенный выше вариант ускоряет вычисления примерно в 1.5 раза.
# Основным источником погрешности является точность np.pi.
import numpy as np
import matplotlib.pyplot as plt
from time import time


def check_accuracy():
    # график точности вычисления экспоненты в зависимости от угла
    phi = 1 + 2 * np.pi * np.arange(10 ** 6)
    plt.plot(phi, np.real(np.exp(1j * phi) - np.exp(1j * phi[0])))
    plt.show()


def check_pi_accuracy():
    # проблема может заключаться в точности определения пи в предыдущем тесте
    # чтобы это устранить, воспользуемся точными данными пи из вики
    pi10power10 = 31415926535.8979323846_2643383279_5028841971_6939937510
    phi0 = 2
    print(np.mod(pi10power10, np.pi))  # ~10**-6
    print(np.exp(1j * (phi0 + pi10power10)) - np.exp(1j * phi0))  # ~10**-6
    # так мы доказали, что порядок ошибки того же порядка, что и погрешность вызываемая неточным определением pi


def check_speed():
    phi_small = np.linspace(-1, 1, 10 ** 7)
    phi_big = 10 ** 10 + phi_small
    for phi in (phi_small, phi_big):
        start = time()
        np.exp(1j * phi)
        print(f'it took {time() - start}')
    # вычисление при больших phi примерно в 2-3 раза дольше


def check_speed2():
    # проверим, что вычисление значений по модулю 2pi даёт примерно ту же точность, ускоряя вычисления
    pi10power10 = 31415926535.8979323846_2643383279_5028841971_6939937510
    phi_small = np.linspace(-1, 1, 10 ** 6)
    two_pi = 2 * np.pi
    phi_big = pi10power10 + phi_small

    # phi_small
    start = time()
    np.exp(1j * phi_small)
    print(f'it took {time() - start}')

    # phi_big
    start = time()
    np.exp(1j * np.mod(phi_big, two_pi))
    print(f'it took {time() - start}')
    # Разница в скорости составила 1.5-2 раза. Это значительно лучше, чем результаты check_speed.
    print(np.exp(1j * np.mod(phi_big[-1], two_pi)) - np.exp(1j * phi_small[-1]))
    # При этом точность осталась на том же уровне.


if __name__ == '__main__':
    check_speed2()