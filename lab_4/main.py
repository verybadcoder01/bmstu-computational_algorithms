import numpy as np
import math
import matplotlib.pyplot as plt

def should_stop(x, dx):
    if np.max(np.abs(dx / x)) < 1e-4:
        return True
    return False


def gauss_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            for k in range(i+1, n):
                if abs(A[k, i]) > 1e-12:
                    A[[i, k]] = A[[k, i]]
                    b[[i, k]] = b[[k, i]]
                    break
        div = A[i, i]
        A[i, :] /= div
        b[i] /= div
        for j in range(i+1, n):
            factor = A[j, i]
            A[j, :] -= factor * A[i, :]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i+1:], x[i+1:])
    return x


def get_jacobian(x, y):
    def f1_x(x, y):
        return 20 / (x - y) - 1

    def f1_y(x, y):
        return -20 / (x - y) - 1

    def f2_x(x, y):
        return 20 * math.cos(0.7 * x - 0.7 * y) * 0.7 + 7

    def f2_y(x, y):
        return -20 * math.cos(0.7 * x - 0.7 * y) * 0.7 + 7
    return np.array([[f1_x(x, y), f1_y(x, y)], [f2_x(x, y), f2_y(x, y)]])


def f1(x, y):
    return 20 * math.log(x - y) - x - y - 6


def f2(x, y):
    return 20 * math.sin(0.7 * x - 0.7 * y) + 7 * x + 7 * y


def get_f_vector(x, y):
    return np.array([f1(x, y), f2(x, y)])


def solve_p1():
    x = 0
    y = -1
    deltas = np.array([10, 10])
    it = 0
    jac = get_jacobian(x, y)
    while not should_stop(np.array([x, y]), deltas):
        f_vec = get_f_vector(x, y)
        deltas = gauss_solve(jac, -f_vec)
        x += deltas[0]
        y += deltas[1]
        it += 1
    print(f"Найденный корень: {x:.6f} {y:.6f}")
    print(f"Значения функций в данной точке: {f1(x, y):.15f}, {f2(x, y):.15f}")
    print(f"Потребовалось {it} итераций")


def compute_p2_integral(b, h):
    def f(t):
        return math.exp(-(t*t) / 2)
    x = h
    sum = 0
    while x < b:
        xprev = x - h
        sum += f((xprev + x) / 2)
        x += h
    return sum * h


def p2_integral_medium(b):
    sign = 1
    if b < 0:
        b = -b
        sign = -1
    h = b / 2
    int_double_prec = compute_p2_integral(b, h/2)
    int_curr_prec = compute_p2_integral(b, h)
    if abs(int_double_prec) < 1e-6:
        return 0
    while abs((int_double_prec - int_curr_prec) / int_double_prec) >= 1e-6:
        int_curr_prec = int_double_prec
        h /= 2
        int_double_prec = compute_p2_integral(b, h/2)
        if abs(int_double_prec) < 1e-6:
            return 0
    return int_curr_prec * sign


def laplace_f(x, y):
    return (2 / math.sqrt(2 * math.pi)) * p2_integral_medium(x) - y


def bisect_p2(y):
    a = 10
    b = 0
    fa = laplace_f(a, y)
    fb = laplace_f(b, y)
    if fa * fb > 0:
        print("Функция не принимает такое значение нигде.")
        return None
    for _ in range(100):
        c = (a + b) / 2
        fc = laplace_f(c, y)
        if abs((b - a) / c) < 1e-6:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0


def solve_p2(y):
    root = bisect_p2(y)
    if root is not None:
        fr = laplace_f(root, y) + y
        print(f"Найденный корень: {root:.6f}, значение функции: {fr:.6f}")


def solve_p3():
    N = 50
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    x_inner = x[1:-1]

    def rhs(x):
        return x**2
    
    y = 1.0 + 2.0 * x
    max_iter = 50
    tol = 1e-4

    for iteration in range(max_iter):
        y_inner = y[1:-1]
        y_left = y[:-2]
        y_right = y[2:]

        F = (y_left - 2*y_inner + y_right) / h**2 - y_inner**3 - rhs(x_inner)
        
        A = np.ones(N-1) / h**2
        C = np.ones(N-1) / h**2
        B = -2.0 / h**2 - 3.0 * y_inner**2
        A[0] = 0.0
        C[-1] = 0.0
        D = -F

        nu = np.zeros(N-1)
        gamma = np.zeros(N-1)

        nu[0] = -C[0] / B[0]
        gamma[0] = D[0] / B[0]
        for i in range(1, N-1):
            denom = B[i] + A[i] * nu[i-1]
            nu[i] = -C[i] / denom
            gamma[i] = (D[i] - A[i] * gamma[i-1]) / denom

        dy = np.zeros(N-1)
        dy[-1] = gamma[-1]
        for i in range(N-3, -1, -1):
            dy[i] = nu[i] * dy[i+1] + gamma[i]

        y[1:-1] += dy
        rel_delta = np.abs(dy / y[1:-1])
        if np.max(rel_delta) < tol:
            print(f"Сошлось за {iteration+1} итераций")
            break
    else:
        print("Достигнуто максимальное число итераций")

    # F_final = (y[:-2] - 2*y[1:-1] + y[2:]) / h**2 - y[1:-1]**3 - rhs(x_inner)
    # print(f"Максимальная невязка: {np.max(np.abs(F_final)):.2e}")

    plt.plot(x, y, 'b-', linewidth=2, label='Численное решение')
    plt.plot(x, 1+2*x, 'r--', label='Начальное приближение')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Решение краевой задачи $y'' - y^3 = x^2$, $y(0)=1$, $y(1)=3$")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    menu = -1
    while menu != 0:
        menu = int(input("Введите действие: 1, 2, 3 или 0 - выход\n"))
        if menu == 1:
            solve_p1()
        elif menu == 2:
            y = float(input("Введите значение функции\n"))
            solve_p2(y)
        elif menu == 3:
            solve_p3()
        else:
            break


if __name__ == '__main__':
    main()
