import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(- x **2/2)
    # return np.sinc(x) * np.sqrt(2*np.pi)

def my_ft(x, f):
    A, L, N = x[0], x[-1] - x[0], len(x)
    w = (np.arange(N) - N // 2) * 2 * np.pi/L
    f_ft = np.fft.fft(f) / np.sqrt(2*np.pi) / N * L * np.exp(- 1j * A * w)
    f_ft = np.concatenate((f_ft[N // 2:], f_ft[:N // 2]))
    return w, f_ft

A, L, N = -10, 20, 10 ** 3
x_arr = np.linspace(A, A+L, N)
f_arr = f(x_arr)

w, f_ft = my_ft(x_arr, f_arr)
plt.plot(w, np.real(f_ft))
plt.plot(w, np.imag(f_ft))
# plt.plot(x_arr, f_arr)
plt.show()
