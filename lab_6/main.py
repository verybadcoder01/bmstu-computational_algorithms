import math
from matplotlib import pyplot as plt

def p1():
    x = [1, 2, 3, 4, 5, 6]
    y = [0.571, 0.889, 1.091, 1.231, 1.333, 1.412]
    n = len(x)
    h = 1

    der1 = [0.0] * n
    for i in range(n - 1):
        der1[i] = (y[i + 1] - y[i]) / h
    der1[n - 1] = (y[n - 1] - y[n - 2]) / h
    der2 = [None] * n
    for i in range(1, n - 1):
        der2[i] = (y[i + 1] - y[i - 1]) / (2 * h)

    der3 = [0.0] * n
    for i in range(n - 2):
        F1 = (y[i + 1] - y[i]) / h
        F2 = (y[i + 2] - y[i]) / (2 * h)
        der3[i] = 2 * F1 - F2
    for i in range(n - 2, n):
        F1_left = (y[i] - y[i - 1]) / h
        F2_left = (y[i] - y[i - 2]) / (2 * h)
        der3[i] = 2 * F1_left - F2_left

    v = [1.0 / xi for xi in x]
    u = [1.0 / yi for yi in y]
    der4 = [0.0] * n
    k = [0.0] * n
    k[0] = (u[1] - u[0]) / (v[1] - v[0])
    for i in range(1, n - 1):
        k[i] = (u[i + 1] - u[i - 1]) / (v[i + 1] - v[i - 1])
    k[n - 1] = (u[n - 1] - u[n - 2]) / (v[n - 1] - v[n - 2])

    for i in range(n):
        der4[i] = k[i] * (y[i] ** 2) / (x[i] ** 2)

    der5 = [None] * n
    for i in range(1, n - 1):
        der5[i] = (y[i + 1] - 2 * y[i] + y[i - 1]) / (h ** 2)

    def fmt(val):
        if val is None:
            return "   —   "
        return f"{val:7.4f}"

    header = f"{'x':>6} {'y':>8} {'(1)':>8} {'(2)':>8} {'(3)':>8} {'(4)':>8} {'(5)':>8}"
    print(header)
    print("-" * len(header))
    for i in range(n):
        print(
            f"{x[i]:6} {y[i]:8.3f} {fmt(der1[i])} {fmt(der2[i])} {fmt(der3[i])} {fmt(der4[i])} {fmt(der5[i])}")


def p2():
    alpha, beta, gamma = map(float, input(
        "Пожалуйста, введите альфа, бета, гамма\n").split())
    N = 100
    h = 1 / N
    Bconst = 4 * h * h - 2
    def x(i):
        return i * h
    
    A = [0.0] * (N + 1)
    B = [0.0] * (N + 1)
    C = [0.0] * (N + 1)
    F = [0.0] * (N + 1)
    for i in range(0, N + 1):
        A[i] = 1 + h * x(i) ** 2
        B[i] = Bconst
        C[i] = 1 - h * x(i) ** 2
        F[i] = h * h * (2 * x(i) + math.exp(-x(i)))
    
    eta = [0.0] * (N + 1)
    nu = [0.0] * (N + 1)
    u = [0.0] * (N + 1)
    eta[0] = -2 / Bconst
    nu[0] = (h * h + 2 * h * alpha) / Bconst
    for i in range(1, N):
        eta[i] = -C[i] / (A[i] * eta[i - 1] + B[i])
        nu[i] = (F[i] - A[i] * nu[i - 1]) / (A[i] * eta[i - 1] + B[i])

    u[N] = (F[N] - 2 * h * gamma * C[N] - 2 * nu[N - 1]) / (2 * eta[N - 1] + B[N] + 2 * h * beta * C[N])
    for i in range(N - 1, -1, -1):
        u[i] = eta[i] * u[i + 1] + nu[i]
    
    x_vals = [x(i) for i in range(N + 1)]
    plt.plot(x_vals, u, label='u(x)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Решение краевой задачи методом прогонки')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    p1()
    p2()


if __name__ == '__main__':
    main()
