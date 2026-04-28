import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def polyfit_1d(x, y, weights, n):
    N = len(x)
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    for i in range(N):
        xi = x[i]
        yi = y[i]
        wi = weights[i]
        powers = np.array([xi**k for k in range(2*n+1)])
        for p in range(n+1):
            for q in range(n+1):
                A[p, q] += wi * powers[p+q]
            b[p] += wi * yi * powers[p]
    return gauss_solve(A, b)


def polyval_1d(coeff, x):
    return sum(c * x**k for k, c in enumerate(coeff))


def polyfit_2d(x, y, z, weights, degree=1):
    N = len(x)
    if degree == 1:
        def basis(xi, yi): return [1.0, xi, yi]
        n_coeff = 3
    elif degree == 2:
        def basis(xi, yi): return [1.0, xi, yi, xi**2, xi*yi, yi**2]
        n_coeff = 6
    else:
        raise ValueError("Only degree 1 or 2")

    A = np.zeros((n_coeff, n_coeff))
    b = np.zeros(n_coeff)
    for i in range(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        wi = weights[i]
        phi = np.array(basis(xi, yi))
        for p in range(n_coeff):
            for q in range(n_coeff):
                A[p, q] += wi * phi[p] * phi[q]
            b[p] += wi * zi * phi[p]
    return gauss_solve(A, b)


def polyval_2d(coeff, x, y, degree=1):
    if degree == 1:
        return coeff[0] + coeff[1]*x + coeff[2]*y
    else:
        return coeff[0] + coeff[1]*x + coeff[2]*y + coeff[3]*x**2 + coeff[4]*x*y + coeff[5]*y**2


def generate_1d_data(N=2, noise=0.5):
    # y = sin(2*pi*x) + шум
    np.random.seed(42)
    x = np.random.uniform(0, 2*np.pi, N)
    x.sort()
    y_true = np.sin(x)
    y = y_true + np.random.normal(0, noise, N)
    rho = np.ones(N)
    return x, y, rho


def generate_2d_data(N=30, noise=0.1):
    # [0,1]x[0,1], z = sin(pi*x)*cos(pi*y) + шум
    np.random.seed(57)
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    z_true = 1 + 2*x + 3*y
    # z = z_true + np.random.normal(0, noise, N)
    rho = np.ones(N)
    return x, y, z_true, rho


x3 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
y3 = np.array([10.5, 1.6, 0.55, 0.26, 0.15, 0.08])
weights3 = np.ones_like(x3)


def fit_power_law(x, y):
    # ln f = ln a + b ln x
    lnx = np.log(x)
    lny = np.log(y)
    coeff = polyfit_1d(lnx, lny, np.ones_like(x), 1)
    a = np.exp(coeff[0])
    b = coeff[1]
    return lambda x: a * x**b, (a, b)


def fit_exp(x, y):
    # ln f = ln a + bx
    coeff = polyfit_1d(x, np.log(y), np.ones_like(x), 1)
    a = np.exp(coeff[0])
    b = coeff[1]
    return lambda x: a * np.exp(b*x), (a, b)


def fit_hyperbolic(x, y):
    # x -> 1/x
    invx = 1.0 / x
    coeff = polyfit_1d(invx, y, np.ones_like(x), 1)
    a, b = coeff
    return lambda x: a + b/x, (a, b)


def fit_rational(x, y):
    # y -> 1/y
    invy = 1.0 / y
    coeff = polyfit_1d(x, invy, np.ones_like(x), 1)
    A, B = coeff
    return lambda x: 1.0 / (A + B*x*x), (A, B)


def sqrt_mean_err(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def select_best_function(x, y):
    models = {
        "Степенная a*x^b": fit_power_law,
        "Экспоненциальная a*e^(bx)": fit_exp,
        "Гиперболическая a + b/x": fit_hyperbolic,
        "Дробно-линейная 1/(A+Bx^2)": fit_rational
    }
    best_name = None
    best_func = None
    best_error = float('inf')
    results = {}
    for name, fit_func in models.items():
        f, params = fit_func(x, y)
        y_pred = f(x)
        err = sqrt_mean_err(y, y_pred)
        results[name] = (err, params)
        if err < best_error:
            best_error = err
            best_name = name
            best_func = f
    return best_name, best_func, best_error, results


def solve_ode(m, n_colloc=30):
    def u0(x):
        return 1 - x

    def uk(x, k):
        return (x**k) * (1 - x)

    def duk(x, k):
        return k*x**(k-1) - (k+1)*x**k

    def d2uk(x, k):
        if k == 0:
            return 0
        return k*(k-1)*x**(k-2) - (k+1)*k*x**(k-1)

    colloc_points = np.linspace(0, 1, n_colloc+2)[1:-1]
    R0 = 1 - 4*colloc_points

    A = np.zeros((len(colloc_points), m))
    for i, xp in enumerate(colloc_points):
        for k in range(1, m+1):
            A[i, k-1] = d2uk(xp, k) + xp * duk(xp, k) + uk(xp, k)

    AtA = A.T @ A
    AtR0 = A.T @ R0
    C = np.linalg.solve(AtA, -AtR0)

    def y_sol(x):
        val = u0(x)
        for k in range(1, m+1):
            val += C[k-1] * uk(x, k)
        return val
    return y_sol, C


def interactive_weight_editor(x, y, current_weights, poly_degree=1):
    weights = current_weights.copy()
    while True:
        print("\n--- Текущие веса точек ---")
        print("idx\tx\t\ty\t\tweight")
        for i in range(len(x)):
            print(f"{i}\t{x[i]:.4f}\t{y[i]:.4f}\t{weights[i]:.4f}")
        print("--------------------------------")
        print("Команды: 'edit <индекс> <новый_вес>'  'plot'  'degree <n>'  'exit'")
        cmd = input("> ").strip().lower()
        if cmd == 'exit':
            break
        elif cmd == 'plot':
            coeff = polyfit_1d(x, y, weights, poly_degree)
            x_plot = np.linspace(min(x), max(x), 200)
            y_plot = polyval_1d(coeff, x_plot)
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, label='Точки', color='red')
            plt.plot(x_plot, y_plot, label=f'Полином {poly_degree}-й степени')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(
                f'Аппроксимация с текущими весами (степень {poly_degree})')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif cmd.startswith('degree'):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                new_deg = int(parts[1])
                if new_deg >= 0 and new_deg < len(x):
                    poly_degree = new_deg
                    print(f"Степень полинома изменена на {poly_degree}")
                else:
                    print(f"Степень должна быть от 0 до {len(x)-1}")
            else:
                print("Использование: degree <n>")
        elif cmd.startswith('edit'):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    idx = int(parts[1])
                    new_w = float(parts[2])
                    if 0 <= idx < len(x):
                        if new_w < 0:
                            print("Вес не может быть отрицательным, установлен 0")
                            new_w = 0
                        weights[idx] = new_w
                        print(f"Вес точки {idx} изменён на {new_w}")
                    else:
                        print(f"Индекс должен быть от 0 до {len(x)-1}")
                except ValueError:
                    print("Некорректные числа")
            else:
                print("Использование: edit <индекс> <новый_вес>")
        else:
            print("Неизвестная команда")
    return weights


def main():
    print("1. Одномерная аппроксимация")
    x1d, y1d, rho1d = generate_1d_data(N=3, noise=0.15)
    print("Сгенерировано 12 точек. Веса по умолчанию = 1.")
    coeff1 = polyfit_1d(x1d, y1d, rho1d, 1)
    coeff2 = polyfit_1d(x1d, y1d, rho1d, 2)

    x_plot = np.linspace(min(x1d), max(x1d), 200)
    y_plot1 = polyval_1d(coeff1, x_plot)
    y_plot2 = polyval_1d(coeff2, x_plot)

    plt.figure(figsize=(10, 5))
    plt.scatter(x1d, y1d, label='Исходные точки', color='red')
    plt.plot(x_plot, y_plot1,
             label=f'Полином 1-й степени: {coeff1[0]:.2f} + {coeff1[1]:.2f}x')
    plt.plot(x_plot, y_plot2, label=f'Полином 2-й степени')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Одномерная аппроксимация (веса = 1)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nДемонстрация влияния весов:")
    rho_modified = np.ones_like(x1d)
    rho_modified[-3:] = 10.0
    coeff1_mod = polyfit_1d(x1d, y1d, rho_modified, 1)
    y_plot1_mod = polyval_1d(coeff1_mod, x_plot)

    plt.figure(figsize=(10, 5))
    plt.scatter(x1d, y1d, label='Точки', color='red')
    plt.plot(x_plot, y_plot1, label='Прямая (веса=1)', linestyle='--')
    plt.plot(x_plot, y_plot1_mod,
             label='Прямая (большие веса у последних точек)', linestyle='-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Влияние весов на прямую')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Режим изменения весов ---")
    try:
        deg = int(input("Введите степень полинома: "))
    except:
        deg = 1
    initial_weights = np.ones_like(x1d)
    interactive_weight_editor(x1d, y1d, initial_weights, poly_degree=deg)

    print("\n2. Двумерная аппроксимация")
    x2d, y2d, z2d, rho2d = generate_2d_data(N=30, noise=0.1)

    coeff2d_1 = polyfit_2d(x2d, y2d, z2d, rho2d, degree=1)
    for i in coeff2d_1:
        print(i)
    coeff2d_2 = polyfit_2d(x2d, y2d, z2d, rho2d, degree=2)

    x_grid = np.linspace(0, 1, 30)
    y_grid = np.linspace(0, 1, 30)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Z_pred1 = polyval_2d(coeff2d_1, Xg, Yg, degree=1)
    Z_pred2 = polyval_2d(coeff2d_2, Xg, Yg, degree=2)
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x2d, y2d, z2d, c='red', s=20, label='Точки')
    ax1.set_title('Исходные точки')
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(Xg, Yg, Z_pred1, alpha=0.7, cmap='viridis')
    ax2.scatter(x2d, y2d, z2d, c='red', s=10)
    ax2.set_title('Полином 1-й степени')
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(Xg, Yg, Z_pred2, alpha=0.7, cmap='plasma')
    ax3.scatter(x2d, y2d, z2d, c='red', s=10)
    ax3.set_title('Полином 2-й степени')
    plt.show()

    print("\n3. Выбор оптимальной аппроксимирующей функции из ряда вариантов")
    best_name, best_func, best_err, results = select_best_function(x3, y3)
    print("Таблица точек:")
    for xi, yi in zip(x3, y3):
        print(f"  x={xi:.1f}  y={yi:.3f}")
    print("\nРезультаты сравнения:")
    for name, (err, params) in results.items():
        print(f"  {name:25}  Отклонение = {err:.5f}  (параметры: {params[0]:.5f} {params[1]:.5f})")
    print(f"\nЛучшая функция: {best_name} с отклонением = {best_err:.5f}")

    x_plot3 = np.linspace(0.4, 3.2, 100)
    y_best = best_func(x_plot3)
    plt.figure(figsize=(8, 5))
    plt.scatter(x3, y3, label='Исходные точки', color='red', s=50)
    plt.plot(x_plot3, y_best, label=f'Лучшая: {best_name}', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Оптимальная аппроксимация')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n4. Приближённое решение краевой задачи для ОДУ")
    print("Уравнение: y'' + x y' + y = 2x,  y(0)=1, y(1)=0")
    y_sol2, C2 = solve_ode(m=2)
    y_sol3, C3 = solve_ode(m=3)
    x_plot4 = np.linspace(0, 1, 100)
    y_vals2 = y_sol2(x_plot4)
    y_vals3 = y_sol3(x_plot4)
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot4, y_vals2, label=f'm=2, коэффициенты: {C2[0]:.5f} {C2[1]:.5f}')
    plt.plot(x_plot4, y_vals3, label=f'm=3, коэффициенты: {C3[0]:.5f} {C3[1]:.5f} {C3[2]:.5f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Приближённое решение ОДУ')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
