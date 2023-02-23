import numpy as np

def puas_coef(x, n):
    if n < 3:
        if n == 0:
            return 1
        if n == 1:
            return x
        if n == 2:
            return x * x / 2
    else:
        return np.exp(- 1 / 12 / n)/np.sqrt(2*np.pi *n) * (np.e * x/n) ** n

def my_minus_exp(x):
    n_mean = np.int_(x)+1
    n_max = 2 ** (np.int_(np.log2(np.round(x + 3 * x ** (3 / 4)))) + 1)
    n_arr = np.arange(0, n_max)
    f_n0 = 1 - 2 * np.mod(n_arr, 2)

    row = np.stack((n_arr, f_n0)).T
    new_row = partial_simplify(x, row, n_mean)
    # new_new_row = partial_simplify(x, new_row, n_mean // 2)
    return np.array((sum_row(x, new_row),sum_row(x, row)))  * np.exp(x) - 1

def partial_simplify(x, row, n_mean):
    new_row = np.zeros((len(row) // 2, 2))
    for i in range(len(new_row)):
        if 2 * i >= n_mean:
            new_row[i] = np.array((2 * i + 1, row[2 * i + 1][1] + row[2 * i][1] * (2 * i + 1) / x))
        else:
            new_row[i] = np.array((2 * i, row[2 * i][1] + row[2 * i + 1][1] * x / (2 * i + 1)))
    return new_row

def sum_row(x, row):
    sum_ = 0
    for n, f in row:
        sum_ += puas_coef(x, n) * f
    return sum_


if __name__ == '__main__':
    print(my_minus_exp(6))
