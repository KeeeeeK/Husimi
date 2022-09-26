from time import time
import numba as nb


def check_simple_compilation_time():
    N = 1000

    def func_gen(i):
        def func(a0, a1, a2, a4):
            return a0 + a1 / a2 * a4 + i

        return func

    start = time()
    nb.vectorize(nb.njit(func_gen(0)))
    print(f'generated 1 func in {time() - start}')

    start = time()
    for i in range(N):
        nb.vectorize(nb.njit(func_gen(i)))
    print(f'generated {N} funcs in {time() - start}')


if __name__ == '__main__':
    check_simple_compilation_time()
