import numpy as np
import matplotlib.pyplot as plt

t0 = 14e-6
tk = 450e-6
T0 = 5400
px = 0.04
Tx = 300
R = 0.25
l = 12
tau = 3*1e-6

deg_t = 3
deg_T = 1
deg_p = 1

p_min = 0.3
p_max = 2.5
tol_p = 1e-4

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


class NewtonInterpolant:
    def __init__(self, x, y):
        idx = np.argsort(x)
        self.x = np.array(x)[idx]
        self.y = np.array(y)[idx]

    def evaluate(self, x, degree=None):
        if degree is None:
            degree = len(self.x) - 1
        n = min(degree + 1, len(self.x))
        idx = select_nodes(x, self.x, n-1)
        x_loc = self.x[idx]
        y_loc = self.y[idx]
        return newton_eval(x_loc, y_loc, x, n-1)

class BivariateNewtonInterpolant:
    def __init__(self, T_vals, p_vals, f_2d, deg_T, deg_p):
        self.T_nodes = np.array(T_vals)
        self.p_nodes = np.array(p_vals)
        self.f_2d = np.array(f_2d)
        idxT = np.argsort(self.T_nodes)
        self.T_nodes = self.T_nodes[idxT]
        self.f_2d = self.f_2d[idxT, :]
        idxP = np.argsort(self.p_nodes)
        self.p_nodes = self.p_nodes[idxP]
        self.f_2d = self.f_2d[:, idxP]
        self.deg_T = deg_T
        self.deg_p = deg_p

    def evaluate(self, T, p):
        idxT = select_nodes(T, self.T_nodes, self.deg_T)
        idxP = select_nodes(p, self.p_nodes, self.deg_p)
        T_loc = self.T_nodes[idxT]
        p_loc = self.p_nodes[idxP]

        f_at_T = []
        for i in idxT:
            f_row = self.f_2d[i][idxP]          
            f_i = newton_eval(p_loc, f_row, p, self.deg_p)
            f_at_T.append(f_i)

        return newton_eval(T_loc, f_at_T, T, self.deg_T)

def read_1d_table(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[-2]))
                y.append(float(parts[-1]))
    return np.array(x), np.array(y)

def read_2d_table(filename):
    T = []
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                T.append(float(parts[0]))
                data.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(T), np.array(data)

def solve_p(T, Nh_interp, const, p_min, p_max, tol=1e-8, max_iter=100):
    f = lambda p: Nh_interp.evaluate(T, p) - const
    f_min = f(p_min)
    f_max = f(p_max)

    if f_min * f_max > 0:
        p_min_try, p_max_try = p_min * 0.5, p_max * 2.0
        f_min_try = f(p_min_try)
        f_max_try = f(p_max_try)
        if f_min_try * f_max_try <= 0:
            p_min, p_max = p_min_try, p_max_try
            f_min, f_max = f_min_try, f_max_try
        else:
            raise ValueError(f"Не удалось локализовать корень при T={T:.1f}: "
                             f"f({p_min})={f_min:.3e}, f({p_max})={f_max:.3e}")

    for _ in range(max_iter):
        p_mid = (p_min + p_max) / 2
        f_mid = f(p_mid)
        if abs(f_mid) < tol:
            return p_mid
        if f_min * f_mid < 0:
            p_max = p_mid
            f_max = f_mid
        else:
            p_min = p_mid
            f_min = f_mid
    return (p_min + p_max) / 2

def main():
    print("Загрузка данных")
    t_vals, I_vals = read_1d_table('It.txt')
    T_c, data_c = read_2d_table('cTp.txt')
    T_sigma, data_sigma = read_2d_table('sigmaTp.txt')
    T_q, data_q = read_2d_table('qTp.txt')
    T_Nh, data_Nh = read_2d_table('NhTp.txt')

    print("Построение интерполянтов")
    I_interp = NewtonInterpolant(t_vals, I_vals)
    c_interp = BivariateNewtonInterpolant(T_c, [0.5, 1.5, 2.5], data_c, deg_T, deg_p)
    sigma_interp = BivariateNewtonInterpolant(T_sigma, [0.5, 1.5, 2.5], data_sigma, deg_T, deg_p)
    q_interp = BivariateNewtonInterpolant(T_q, [0.5, 1.5, 2.5], data_q, deg_T, deg_p)
    Nh_interp = BivariateNewtonInterpolant(T_Nh, [0.5, 1.5, 2.5], data_Nh, deg_T, deg_p)

    const_p = 7.242e4 * px / Tx
    S = np.pi * R**2

    N = int((tk - t0) / tau)
    t = np.linspace(t0, tk, N+1)

    T_arr = np.zeros(N+1)
    p_arr = np.zeros(N+1)
    sigma_arr = np.zeros(N+1)
    q_arr = np.zeros(N+1)
    c_arr = np.zeros(N+1)
    Rd_arr = np.zeros(N+1)
    Fr_arr = np.zeros(N+1)

    T_arr[0] = T0
    print(f"Вычисление начального давления при T0 = {T0} K")
    p_arr[0] = solve_p(T0, Nh_interp, const_p, p_min, p_max, tol_p)
    I0 = I_interp.evaluate(t[0], deg_t)
    j0 = I0 / S
    sigma_arr[0] = sigma_interp.evaluate(T0, p_arr[0])
    q_arr[0] = q_interp.evaluate(T0, p_arr[0])
    c_arr[0] = c_interp.evaluate(T0, p_arr[0])
    Rd_arr[0] = l / (np.pi * sigma_arr[0] * R**2)
    Fr_arr[0] = q_arr[0] * R / 2

    print("Интегрирование ОДУ")
    for n in range(N):
        T_n = T_arr[n]
        p_n = p_arr[n]
        t_n = t[n]
        sigma_n = sigma_arr[n]
        q_n = q_arr[n]
        c_n = c_arr[n]
        I_n = I_interp.evaluate(t_n, deg_t)
        j_n = I_n / S
        phi_n = (j_n**2 / sigma_n - q_n) / c_n
        T_mid = T_n + tau * phi_n
        p_mid = solve_p(T_mid, Nh_interp, const_p, p_min, p_max, tol_p)
        t_mid = t_n + tau/2
        I_mid = I_interp.evaluate(t_mid, deg_t)
        j_mid = I_mid / S

        sigma_mid = sigma_interp.evaluate(T_mid, p_mid)
        q_mid = q_interp.evaluate(T_mid, p_mid)
        c_mid = c_interp.evaluate(T_mid, p_mid)

        phi_mid = (j_mid**2 / sigma_mid - q_mid) / c_mid
        T_next = T_n + tau * phi_mid
        T_arr[n+1] = T_next

        p_arr[n+1] = solve_p(T_next, Nh_interp, const_p, p_min, p_max, tol_p)
        sigma_arr[n+1] = sigma_interp.evaluate(T_next, p_arr[n+1])
        q_arr[n+1] = q_interp.evaluate(T_next, p_arr[n+1])
        c_arr[n+1] = c_interp.evaluate(T_next, p_arr[n+1])
        Rd_arr[n+1] = l / (np.pi * sigma_arr[n+1] * R**2)
        Fr_arr[n+1] = q_arr[n+1] * R / 2

        if (n+1) % 50 == 0:
            print(f"  шаг {n+1}/{N}, t = {t[n+1]:.2e} c, T = {T_next:.1f} K")

    print("Сохранение результатов")
    out_filename = "results.csv"
    header = "t(s),T(K),p(MPa),sigma(1/(Ohm*cm)),q(W/cm3),Rd(Ohm),Fr(W/cm)"
    data_out = np.column_stack((t, T_arr, p_arr, sigma_arr, q_arr, Rd_arr, Fr_arr))
    np.savetxt(out_filename, data_out, delimiter=',', header=header, comments='')
    print(f"Результаты записаны в {out_filename}")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(t*1e6, T_arr, 'b-')
    plt.xlabel('t, мкс')
    plt.ylabel('T, K')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(t*1e6, p_arr, 'r-')
    plt.xlabel('t, мкс')
    plt.ylabel('p, МПа')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(t*1e6, sigma_arr, 'g-')
    plt.xlabel('t, мкс')
    plt.ylabel('σ, 1/(Ом·см)')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(t*1e6, q_arr, 'm-')
    plt.xlabel('t, мкс')
    plt.ylabel('q, Вт/см³')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(t*1e6, Rd_arr, 'c-')
    plt.xlabel('t, мкс')
    plt.ylabel('R_d, Ом')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(t*1e6, Fr_arr, 'y-')
    plt.xlabel('t, мкс')
    plt.ylabel('F_r, Вт/см')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
