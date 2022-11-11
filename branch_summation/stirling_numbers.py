import numpy as np


def stirling_3(n_max, k_max):
    """Генерирует таблицу размера n_max+1 на k_max+1, в которой расположены числа S_3(n, k)"""
    if n_max < 2:
        return _small_stirling_3_args(n_max, k_max)
    res = np.zeros((n_max + 1, k_max + 1), dtype=np.int_)
    res[0, 0] = 1
    for n in range(1, n_max - 1):
        for k in range(1, k_max):
            res[n + 1, k] = k * res[n, k] + int(n * (n - 1) / 2) * res[n - 2, k - 1]
    return res


def _small_stirling_3_args(n_max, k_max):
    if n_max == 0:
        return np.array([[1] + [0] * k_max])
    if n_max == 1:
        if k_max != 0:
            return np.array([[1] + [0] * k_max, [0] * (k_max + 1)])
        else:
            return np.array([[1], [0]])


if __name__ == '__main__':
    # print(stirling_3(10, 10))
    n_max, j_max = 6, 6
    arr = stirling_3(2*n_max + 2 * j_max, 2*j_max)
    # print(arr)
    res = np.zeros((n_max + 1, j_max + 1), dtype=np.int_)
    for n in range(n_max + 1):
        for j in range(j_max + 1):
            res[n][j] = arr[2*n + 2 * j][2*j]
    print(res[2])
