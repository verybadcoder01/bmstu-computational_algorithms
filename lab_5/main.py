import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter

DATA_FILE = "23-04-2026-подинтеграл_функция_лаб_5.txt"
x_values = np.array([])
y_values = []
z_values = []

def get_N_list(method, max_N=30):
    if method == gauss_method:
        return list(range(4, max_N + 1))
    elif method == simpson_method:
        return [n for n in range(4, max_N + 1) if n % 2 == 0]
    else:
        return list(range(2, max_N + 1))

def load_table_data(filename):
    global x_values, y_values, z_values
    with open(filename, 'r', encoding='utf-16') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    x_values = np.array([float(xi) for xi in header[1:]])

    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        y = float(parts[0])
        y_values.append(y)
        z_row = [float(v) for v in parts[1:]]
        z_values.append(z_row)

    y_values = np.array(y_values)
    z_values = np.array(z_values)
    return x_values, y_values, z_values

def select_nodes(x, x_nodes, degree):
    n = degree + 1
    if n >= len(x_nodes):
        return np.arange(len(x_nodes))
    i = np.searchsorted(x_nodes, x)
    if i == 0:
        return np.arange(n)
    elif i == len(x_nodes):
        return np.arange(len(x_nodes)-n, len(x_nodes))
    else:
        start = i - n//2
        if start < 0:
            start = 0
        elif start + n > len(x_nodes):
            start = len(x_nodes) - n
        return np.arange(start, start+n)

def newton_eval(x_nodes, y_nodes, x, degree):
    n = degree + 1
    coef = np.array(y_nodes, dtype=float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_nodes[i] - x_nodes[i-j])
    res = coef[-1]
    for i in range(n-2, -1, -1):
        res = res * (x - x_nodes[i]) + coef[i]
    return res


def evaluate(x, y, x_vals, y_vals, f_vals, deg_x, deg_y):
    idxX = select_nodes(x, x_vals, deg_x)
    idxY = select_nodes(y, y_vals, deg_y)
    x_loc = x_vals[idxX]
    y_loc = y_vals[idxY]

    f_at_X = []
    for i in idxX:
        f_col = f_vals[idxY, i]          
        f_i = newton_eval(y_loc, f_col, y, deg_y)
        f_at_X.append(f_i)

    return newton_eval(x_loc, f_at_X, x, deg_x)

def interp_func(x, y):
    global x_values, y_values, z_values
    return evaluate(x, y, x_values, y_values, z_values, 3, 3)


def func_p1(x, k):
    return abs(x) ** k


def trapezoid_method(a, b, N, func, *args):
    h = (b - a) / N
    sum = (func(a, *args) + func(b, *args)) / 2
    for i in range(1, N):
        sum += func(a + i * h, *args)
    return sum * h


def simpson_method(a, b, N, func, *args):
    if N % 2 == 1:
        raise Exception("N must be even for Simpson's method")
    h = (b - a) / N
    sum = 0
    for i in range(0, N // 2):
        sum += func(a + 2 * i * h, *args)
        sum += 4 * func(a + h * (2 * i + 1), *args)
        sum += func(a + h * (2 * i + 2), *args)
    return sum * h / 3


def gauss_method_3(func, *args):
    t1 = -math.sqrt(3/5)
    t2 = 0
    t3 = math.sqrt(3/5)
    return 1/9 * (5 * func(t1, *args) + 8 * func(t2, *args) + 5 * func(t3, *args))


def gauss_method(a, b, N, func, *args):
    factor = (b - a) / 2
    l_roots = legandre_roots(N)
    A = gauss_get_weights(l_roots)
    l_roots = factor * np.array(l_roots) + (a + b) / 2
    f_vals = np.array([func(x, *args) for x in l_roots])
    return factor * np.dot(A, f_vals)


def integrate(a, b, func, method, fixed_N=None):
    sign = 1
    if b < a:
        a, b = b, a
        sign = -1
        
    if fixed_N is not None:
        return sign * method(a, b, fixed_N, func)
    
    integral_eps = 1e-4
    N = 4
    int_cur_prec = method(a, b, N, func)
    int_double_prec = method(a, b, N * 2, func)
    while abs(int_double_prec) > integral_eps and abs((int_double_prec - int_cur_prec) / int_double_prec) >= integral_eps:
        int_cur_prec = int_double_prec
        N *= 2
        int_double_prec = method(a, b, N * 2, func)
    # print(f"Выбрано узлов: {N}")
    return int_cur_prec * sign


def legandre_p(n, x):
    if n == 0:
        return 1
    if n == 1:
        return x
    p0 = 1
    p1 = x
    for m in range(1, n):
        p2 = ((2 * m + 1) * x * p1 - m * p0) / (m + 1)
        p0, p1 = p1, p2
    return p1


def legandre_roots(n):
    if n == 0:
        return []
    if n == 1:
        return [0]
    roots_prev = [0]
    eps = 1e-6
    for m in range(2, n + 1):
        intervals = [-1] + roots_prev + [1]
        new_roots = []
        for i in range(len(intervals) - 1):
            a = intervals[i]
            b = intervals[i + 1]
            fa = legandre_p(m, a)
            fb = legandre_p(m, b)
            if fa * fb > 0:
                continue
            c = (a + b) / 2
            while abs(c) > eps and abs((b - a) / c) > eps:
                c = (a + b) / 2
                fc = legandre_p(m, c)
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            new_roots.append((a + b) / 2)
        roots_prev = new_roots
    return roots_prev


def gauss_get_weights(l_roots):
    n = len(l_roots)
    coeffs = np.zeros((n, n))
    consts = np.array([2/(k+1) if k % 2 == 0 else 0 for k in range(n)])
    for k in range(n):
        coeffs[k, :] = np.array([t**k for t in l_roots])
    weights = np.linalg.solve(coeffs, consts)
    return weights


def inner_integral(x, a, b, method, N_inner=None):
    lower = a * x ** 2
    upper = b * x ** 2
    def fy(y):
        return interp_func(x, y)
    result =  integrate(lower, upper, fy, method, fixed_N=N_inner)
    return result

def integral_p2(a, b, c, d, method_outer, method_inner, N_outer=None, N_inner=None):
    def G(x):
        return inner_integral(x, a, b, method_inner, N_inner)
    return integrate(c, d, G, method_outer, fixed_N=N_outer)


def main():
    action = int(input("Введите номер задания: 1 или 2\n"))
    if action == 1:
        trMk1 = trapezoid_method(-1, 1, 2, func_p1, 1)
        sMk1 = simpson_method(-1, 1, 2, func_p1, 1)
        gMk1 = gauss_method_3(func_p1, 1)
        trueValk1 = 1
        print(f"Аналитически вычисленное значение I = {trueValk1}")
        print("              | Метод трапеций | Метод Симпсона | Метод Гаусса | ")
        print(
            f"k = 1         |    {trMk1:.6f}    |    {sMk1:.6f}    |   {gMk1:.6f}   |")
        print(
            f"Погрешность   |    {trMk1 - trueValk1:.6f}    |    {sMk1 - trueValk1:.6f}   |   {gMk1 - trueValk1:.6f}  |")
        print(
            f"Относительная |    {abs(trMk1 - trueValk1) / trueValk1:.6f}    |    {abs(sMk1 - trueValk1) / trueValk1:.6f}    |   {abs(gMk1 - trueValk1) / trueValk1:.6f}   |")
        trMk2 = trapezoid_method(-1, 1, 2, func_p1, 2)
        sMk2 = simpson_method(-1, 1, 2, func_p1, 2)
        gMk2 = gauss_method_3(func_p1, 2)
        trueValk2 = 2/3
        print(f"Аналитически вычисленное значение I = {trueValk2:.6f}")
        print(
            f"k = 2         |    {trMk2:.6f}    |    {sMk2:.6f}    |   {gMk2:.6f}   |")
        print(
            f"Погрешность   |    {trMk2 - trueValk2:.6f}    |    {sMk2 - trueValk2:.6f}    |   {gMk2 - trueValk2:.6f}   |")
        print(
            f"Относительная |    {abs(trMk2 - trueValk2) / trueValk2:.6f}    |    {abs(sMk2 - trueValk2) / trueValk2:.6f}    |   {abs(gMk2 - trueValk2) / trueValk2:.6f}   |")
    elif action == 2:
        load_table_data(DATA_FILE)
        c = float(input("Введите нижний предел по x, c = "))
        d = float(input("Введите верхний предел по x, d = "))
        a_coeff = float(input("Введите коэффициент a (y = a*x^2): "))
        b_coeff = float(input("Введите коэффициент b (y = b*x^2): "))
        print("Выберите метод для внешнего интеграла:")
        print("1 - Симпсон, 2 - Гаусс")
        m_outer = int(input())
        print("Выберите метод для внутреннего интеграла:")
        print("1 - Симпсон, 2 - Гаусс")
        m_inner = int(input())

        method_map = {
            1: simpson_method,
            2: gauss_method
        }
        outer = method_map.get(m_outer)
        inner = method_map.get(m_inner)

        result = integral_p2(a_coeff, b_coeff, c, d, outer, inner)
        print(f"Результат двойного интеграла: {result:.8f}")
        
        FIX_OUTER = 8
        FIX_INNER = 8

        N_outer_list = get_N_list(outer, max_N=20)
        N_inner_list = get_N_list(inner, max_N=20)

        inner_res = []
        for N_in in N_inner_list:
            val = integral_p2(a_coeff, b_coeff, c, d, outer, inner,
                              N_outer=FIX_OUTER, N_inner=N_in)
            inner_res.append(val)

        outer_res = []
        for N_out in N_outer_list:
            val = integral_p2(a_coeff, b_coeff, c, d, outer, inner,
                              N_outer=N_out, N_inner=FIX_INNER)
            outer_res.append(val)

        N_outer_grid = get_N_list(outer, max_N=12)
        N_inner_grid = get_N_list(inner, max_N=12)
        Z = np.zeros((len(N_outer_grid), len(N_inner_grid)))
        for i, N_out in enumerate(N_outer_grid):
            for j, N_in in enumerate(N_inner_grid):
                try:
                    Z[i, j] = integral_p2(a_coeff, b_coeff, c, d, outer, inner,
                                          N_outer=N_out, N_inner=N_in)
                except np.linalg.LinAlgError:
                    Z[i, j] = np.nan

        Xgrid, Ygrid = np.meshgrid(N_inner_grid, N_outer_grid)

        fig = plt.figure(figsize=(14, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(N_inner_list, inner_res, 'bo-', label=f'N_outer = {FIX_OUTER}')
        ax1.set_xlabel('N_inner')
        ax1.set_ylabel('Integral')
        ax1.set_title('Зависимость от N_inner')
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(N_outer_list, outer_res, 'rs-', label=f'N_inner = {FIX_INNER}')
        ax2.set_xlabel('N_outer')
        ax2.set_ylabel('Integral')
        ax2.set_title('Зависимость от N_outer')
        ax2.legend()
        ax2.grid(True)
        
        for ax in [ax1, ax2]:
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='plain')

        ax3 = fig.add_subplot(2, 1, 2, projection='3d')
        surf = ax3.plot_surface(Xgrid, Ygrid, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
        ax3.set_xlabel('N_inner')
        ax3.set_ylabel('N_outer')
        ax3.set_zlabel('Integral')
        ax3.set_title('3D зависимость')
        ax3.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax3.ticklabel_format(axis='z', style='plain')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
