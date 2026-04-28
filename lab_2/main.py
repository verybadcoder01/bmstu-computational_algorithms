import numpy as np

def cubic_spline(x, y):
    N = len(x)
    h = [0]
    for i in range(1, N):
        h.append(x[i] - x[i - 1])
    gamma, nu = [0, 0, 0], [0, 0, 0.5]
    for i in range(3, N + 1):
        B = -2 * (h[i - 2] + h[i - 1])
        A = h[i - 2]
        D = h[i - 1]
        F = -3 * ((y[i - 1] - y[i - 2]) / h[i - 1] -
                  (y[i - 2] - y[i - 3]) / h[i - 2])
        gamma_prev = gamma[-1]
        gamma.append(D / (B - A * gamma_prev))
        nu.append((F + A * nu[-1]) / (B - A * gamma_prev))
    u = [0] * N + [nu[-1]]
    for i in range(N - 1, 0, -1):
        u[i] = gamma[i + 1] * u[i + 1] + nu[i + 1]
    a, b, d = [], [], []
    for i in range(1, N):
        a.append(y[i - 1])
        b.append((y[i] - y[i - 1]) / h[i] - (1/3)*h[i]*(u[i + 1] + 2 * u[i]))
        d.append((u[i + 1] - u[i]) / (3 * h[i]))
    return a, b, u, d


def compute_spline(a, b, c, d, x, pt):
    ind = -1
    for i in range(0, len(x) - 1):
        if x[i] <= pt <= x[i + 1]:
            ind = i
            break
    if ind == -1:
        raise Exception
    h = pt - x[ind]
    return a[ind] + b[ind] * h + c[ind] * h**2 + d[ind] * h**3


def f_check(x, y, z):
    return x**2 + y**2 + z**2


def generate_data(x, y, z):
    nx, ny, nz = len(x), len(y), len(z)
    f = [[[0 for _ in range(nx)] for __ in range(ny)] for ___ in range(nz)]
    for i in range(0, nz):
        for j in range(0, ny):
            for k in range(0, nx):
                f[k][j][i] = f_check(x[k], y[j], z[i])
    return f

def read_data(file):
    data = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(5)]
    with open(file, 'r') as f:
        z = 0
        y = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                z += 1
                y = 0
                continue
            nums = list(map(int, line.split(None)))
            for i in range(len(nums)):
                data[i][y][z] = nums[i]
            y += 1
    return data
            


def spline_interd3d(f, x, y, z, x0, y0, z0):
    ny, nz = len(y), len(z)
    step1 = [[0 for i in range(ny)] for j in range(nz)]
    for j in range(ny):
        for k in range(nz):
            f_x = f[:][j][k]
            a_x, b_x, c_x, d_x = cubic_spline(x, f_x)
            step1[j][k] = compute_spline(a_x, b_x, c_x, d_x, x, x0)
    step2 = [0] * nz
    for k in range(nz):
        f_y = step1[:][k]
        a_y, b_y, c_y, d_y = cubic_spline(y, f_y)
        step2[k] = compute_spline(a_y, b_y, c_y, d_y, y, y0)
    a_z, b_z, c_z, d_z = cubic_spline(z, step2)
    result = compute_spline(a_z, b_z, c_z, d_z, z, z0)
    return result


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


def newton_interp3d(f, x, y, z, x0, y0, z0):
    degs = list(map(int, input("Введите степени полиномов через пробел\n").split()))
    deg_x, deg_y, deg_z = degs[0], degs[1], degs[2]
    nx, ny, nz = len(x), len(y), len(z)
    step1 = [[0 for i in range(ny)] for j in range(nz)]
    for j in range(ny):
        for k in range(nz):
            f_x = f[:][j][k]
            x_nodes_idx = select_nodes(x0, x, deg_x)
            step1[j][k] = newton_eval(np.array(x)[x_nodes_idx], np.array(f_x)[x_nodes_idx], x0, deg_x)
    step2 = [0] * nz
    for k in range(nz):
        f_y = step1[:][k]
        y_nodes_idx = select_nodes(y0, y, deg_y)
        step2[k] = newton_eval(np.array(y)[y_nodes_idx], np.array(f_y)[y_nodes_idx], y0, deg_y)
    z_nodes_idx = select_nodes(z0, z, deg_z)
    result = newton_eval(np.array(z)[z_nodes_idx], np.array(step2)[z_nodes_idx], z0, deg_z)
    return result


def mixed_interp3d(f, x, y, z, x0, y0, z0):
    nx, ny, nz = len(x), len(y), len(z)
    step1 = [[0 for i in range(ny)] for j in range(nz)]
    method_x = int(input("Выберите метод интерполяции по х: 1 - полином Ньютона, 2 - сплайн\n"))
    deg_x = 0
    if method_x == 1:
        deg_x = int(input("Введите степень полинома\n"))
    for j in range(ny):
        for k in range(nz):
            f_x = f[:][j][k]
            if method_x == 1:
                x_nodes_idx = select_nodes(x0, x, deg_x)
                step1[j][k] = newton_eval(np.array(x)[x_nodes_idx], np.array(f_x)[x_nodes_idx], x0, deg_x)
            else:
                a_x, b_x, c_x, d_x = cubic_spline(x, f_x)
                step1[j][k] = compute_spline(a_x, b_x, c_x, d_x, x, x0)
    step2 = [0] * nz
    deg_y = 0
    method_y = int(input("Выберите метод интерполяции по y: 1 - полином Ньютона, 2 - сплайн\n"))
    if method_y == 1:
        deg_y = int(input("Введите степень полинома\n"))
    for k in range(nz):
        f_y = step1[:][k]
        if method_y == 1:
            y_nodes_idx = select_nodes(y0, y, deg_y)
            step2[k] = newton_eval(np.array(y)[y_nodes_idx], np.array(f_y)[y_nodes_idx], y0, deg_y)
        else:
            a_y, b_y, c_y, d_y = cubic_spline(y, f_y)
            step2[k] = compute_spline(a_y, b_y, c_y, d_y, y, y0)
    method_z = int(input("Выберите метод интерполяции по z: 1 - полином Ньютона, 2 - сплайн\n"))
    if method_z == 1:
        deg_z = int(input("Введите степень полинома\n"))
        z_nodes_idx = select_nodes(z0, z, deg_z)
        result = newton_eval(np.array(z)[z_nodes_idx], np.array(step2)[z_nodes_idx], z0, deg_z)
    else:
        a_z, b_z, c_z, d_z = cubic_spline(z, step2)
        result = compute_spline(a_z, b_z, c_z, d_z, z, z0)
    return result


def main():
    method = int(input("Введите способ интерполяции: 1 - полином Ньютона, 2 - сплайн, 3 - смешанный\n"))
    coords = list(map(float, input("Введите координаты точки через пробел\n").split()))
    x0, y0, z0 = coords[0], coords[1], coords[2]
    x, y, z = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]
    f = read_data("data.txt")
    result = 0
    if method == 1:
        result = newton_interp3d(f, x, y, z, x0, y0, z0)
    elif method == 2:
        result = spline_interd3d(f, x, y, z, x0, y0, z0)
    else:
        result = mixed_interp3d(f, x, y, z, x0, y0, z0)
    print(f"Значение в точке {result:.6f}")
    print(f_check(x0, y0, z0))


if __name__ == '__main__':
    main()
