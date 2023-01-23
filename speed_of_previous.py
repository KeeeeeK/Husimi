import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial

n, m = 20, 30
beta = 10

k_arr = np.arange(0, min(n, m))
values_arr = comb(n, k_arr) * comb(m, k_arr) * factorial(k_arr) / (4*beta)**(2*k_arr)

plt.plot(k_arr, values_arr)
plt.show()