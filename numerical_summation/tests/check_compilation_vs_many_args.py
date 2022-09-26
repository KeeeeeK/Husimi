# Данный файл, чтоб сравнить скорость выполнения двух альтернатив:
# 1. Подсчёт серии f(a, b, c, x) при фиксированных a, b, c на основе скомпилированной f
# 2. Подсчёт g(x) = f(a, b, c, x) при фиксированных a, b, c с дополнительным временем на перекомпиляцию функции g
# Результат: время работы функции = время компиляции + время расчётов
# Время компиляции = [0.27, 0.3, 0.38, 0.41][кол-во аргументов]
# Время работы определяется сложностью функции, но второй вариант быстрее на 10%
# А вообще быстрее, конечно, использование чистого numpy
from time import time
import numba as nb
import numpy as np


def check_simple_compilation_time():
    # За сколько производится компиляция функций с различным числом аргументов?
    # Результаты:
    # 1000 компиляций функции с 1 аргументом заняла 27 секунд
    # C 2 аргументами 30 секунд
    # C 3 аргументами 38 секунд
    # C 4 аргументами 41 секунд
    N = 1000
    args = np.array((0, 1, 1, 1))

    def func_gen(i):
        def func(a0, a1, a2, a3):
            return a0 + a1 + a2/a3 + i
        # def func(a0, a1, a2):
        #     return a0 + a1 + a2 + i

        return func

    start = time()
    print(((nb.njit(func_gen(0))))(*args))
    print(f'generated 1 func in {time() - start}')

    start = time()
    for i in range(N):
        (nb.njit(func_gen(i)))(*args)
    print(f'generated {N} funcs in {time() - start}')


def check_var1():
    # 1. Подсчёт серии f(a, b, c, x) при фиксированных a, b, c на основе скомпилированной f
    a, b, c = 1, 1, 1
    x = np.linspace(1, 2, 10 ** 7)

    @nb.vectorize
    def f(a, b, c, x):
        return a+b/c+a*(c-2) + np.exp(x) * a + np.log(x) - b* x
    f(a, b, c, 1)
    start = time()
    (f(a, b, c, x))
    print(f'var 1 works in {time()-start} sec')


def check_var2():
    # 2. Подсчёт g(x) = f(a, b, c, x) при фиксированных a, b, c с дополнительным временем на перекомпиляцию функции g
    a, b, c = 1, 1, 1
    x = np.linspace(1, 2, 10 ** 7)


    @nb.vectorize
    def g(x):
        return a + b / c + a * (c - 2) + np.exp(x) * a + np.log(x) - b * x
    g(1)
    start = time()
    (g(x))
    print(f'var 2 works in {time() - start} sec')

def check_no_numba():
    a, b, c = 1, 1, 1
    x = np.linspace(1, 2, 10 ** 7)
    start = time()
    vals = (a + b / c + a * (c - 2) + np.exp(x) * a + np.log(x) - b * x)
    print(f'no numba works in {time() - start} sec')



if __name__ == '__main__':
    check_var1()
    check_var2()
    check_no_numba()
