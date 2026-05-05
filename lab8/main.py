import numpy as np
import matplotlib.pyplot as plt
import csv

# --- 1. Математична модель та допоміжні функції ---

# Задана трансцендентна функція
def F(x):
    return np.sin(x) - 0.5 * x

# Перша похідна функції
def Fp(x):
    return np.cos(x) - 0.5

# Друга похідна функції
def F2p(x):
    return - np.sin(x)

# Універсальна функція розділених різниць будь-якого порядку
def divided_diff(points, f):
    n = len(points)
    # Базовий випадок: розділена різниця 0-го порядку — це f(x)
    if n == 1:
        return f(points[0])

    # Рекурсивна формула: (f[x1...xn] - f[x0...xn-1]) / (xn - x0)
    numerator = divided_diff(points[1:], f) - divided_diff(points[:-1], f)
    denominator = points[-1] - points[0]
    return numerator / denominator

# --- 2. Робота з даними та виділення коренів ---

# Табуляція та запис у формат CSV
def tabulate_and_save(f, a, b, h, filename):
    x_range = np.arange(a, b + h, h)
    y_range = f(x_range)

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['x', 'y'])
        writer.writeheader()
        for xv, yv in zip(x_range, y_range):
            writer.writerow({'x': f"{xv:.4f}", 'y': f"{yv:.4f}"})

# Зчитування даних
def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
    return x, y

# Пошук інтервалів та визначення характеру функції (зростає/спадає)
def find_root_intervals(x, y):
    approximations = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            behavior = "зростає" if y[i + 1] > y[i] else "спадає"
            # Початкове наближення як середина інтервалу
            x0 = (x[i] + x[i + 1]) / 2
            approximations.append({
                "x0": x0,
                "behavior": behavior,
                "interval": (x[i], x[i + 1])
            })
    return approximations

# --- 3. Однокрокові методи уточнення ---

# Метод простої ітерації
def simple_iteration(f, x0, tau, eps):
    x_curr = x0
    for i in range(1, 1001):
        x_next = x_curr + tau * f(x_curr)

        # Критерії зупинки
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, i

        x_curr = x_next
    return x_curr, 1000

# Метод Ньютона
def newton_method(f, fp, x0, eps):
    x_curr = x0
    for i in range(1, 1001):
        f_val = f(x_curr)
        fp_val = fp(x_curr)

        if abs(fp_val) < 1e-15:
            return None, i

        # Обчислення кроку уточнення
        delta_x = f_val / fp_val
        x_next = x_curr - delta_x

        # Критерії зупинки
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, i

        x_curr = x_next
    return x_curr, 1000

# Метод Чебишева
def chebyshev_method(f, fp, f2p, x0, eps):
    x_curr = x0
    for i in range(1, 1001):
        f_v = f(x_curr)
        fp_v = fp(x_curr)
        f2p_v = f2p(x_curr)

        if abs(fp_v) < 1e-15:
            return None, i

        term_newton = f_v / fp_v
        term_cheb = 0.5 * (f_v ** 2 * f2p_v) / (fp_v ** 3)

        x_next = x_curr - term_newton - term_cheb

        # Критерії зупинки
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, i

        x_curr = x_next
    return x_curr, 1000

# --- 4. Багатокрокові методи та зворотна інтерполяція ---

# Метод хорд
def chord_method(f, x0, x1, eps):
    x_p, x_c = x0, x1  # Попереднє та поточне значення
    for i in range(1, 1001):
        f_p, f_c = f(x_p), f(x_c)
        if abs(f_c - f_p) < 1e-15:
            break

        x_next = x_c - f_c * (x_c - x_p) / (f_c - f_p)

        # Критерії зупинки
        if abs(f(x_next)) < eps and abs(x_next - x_c) < eps:
            return x_next, i
        x_p, x_c = x_c, x_next
    return x_c, 1000

# Метод парабол
def parabola_method(f, x0, x1, x2, eps):
    xn, xn1, xn2 = x2, x1, x0
    for i in range(1, 1001):
        f_n = f(xn)
        # Коефіцієнти через розділені різниці
        diff2 = divided_diff([xn2, xn1, xn], f)
        diff1 = divided_diff([xn1, xn], f)

        w = (xn - xn1) * diff2 + diff1
        discr = w ** 2 - 4 * diff2 * f_n
        sqrt_discr = np.sqrt(discr + 0j)

        # Вибір знаменника з найбільшим модулем
        denom = -w + sqrt_discr if abs(-w + sqrt_discr) > abs(-w - sqrt_discr) else -w - sqrt_discr
        x_next = xn + (2 * f_n) / denom

        # Критерії зупинки
        if abs(f(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next.real if abs(x_next.imag) < 1e-12 else x_next, i
        xn2, xn1, xn = xn1, xn, x_next
    return xn, 1000

# Метод зворотної інтерполяції
def inverse_interpolation(f, x0, x1, x2, eps):
    xn, xn1, xn2 = x2, x1, x0
    for i in range(1, 1001):
        yn, yn1, yn2 = f(xn), f(xn1), f(xn2)

        # Розрахунок трьох доданків формули Лагранжа
        t2 = (yn1 * yn) / ((yn2 - yn1) * (yn2 - yn)) * xn2
        t1 = (yn2 * yn) / ((yn1 - yn2) * (yn1 - yn)) * xn1
        t0 = (yn2 * yn1) / ((yn - yn2) * (yn - yn1)) * xn

        x_next = t2 + t1 + t0

        # Критерії зупинки
        if abs(f(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, i

        xn2, xn1, xn = xn1, xn, x_next
    return xn, 1000

# --- 5. Методи для алгебраїчних рівнянь ---

# Обчислення значення многочлена за заданими коефіцієнтами
def poly_eval(coeffs, x):
    res = 0
    for i, a in enumerate(coeffs):
        res += a * (x ** i)
    return res

# Зберігання коефіцієнти у текстовий файл
def save_coefficients(coeffs, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        # Запис коефіцієнтів через пробіл: "a0 a1 a2 a3"
        file.write(" ".join(map(str, coeffs)))

# Зчитування коефіцієнтів многочлена з файлу
def read_coefficients(filename):
    with open(filename, 'r') as f:
        coeffs = [float(x) for x in f.read().split()]
    return np.array(coeffs)

# Метод Ньютона зі схемою Горнера
def horner_newton(coeffs, x0, eps):
    m = len(coeffs) - 1  # Степінь многочлена
    xn = x0

    for iter_count in range(1, 1001):
        # Обчислення b_i для значення функції b0 = F(xn)
        b = np.zeros(m + 1)
        b[m] = coeffs[m]
        for i in range(m - 1, -1, -1):
            b[i] = coeffs[i] + xn * b[i + 1]

        # Обчислення c_i для значення похідної c1 = F'(xn)
        c = np.zeros(m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1):
            c[i] = b[i] + xn * c[i + 1]

        f_val = b[0]
        fp_val = c[1]

        if abs(fp_val) < 1e-15:
            break

        x_next = xn - f_val / fp_val

        # Критерії зупинки
        if abs(x_next - xn) < eps and abs(f_val) < eps:
            return x_next, iter_count

        xn = x_next
    return xn, 1000

# Метод Ліна для комплексних коренів
def lin_method(a, p0, q0, eps):
    m = len(a) - 1
    p, q = p0, q0

    for iter_count in range(1, 1001):
        b = np.zeros(m + 1)
        # Знаходження коефіцієнтів b_i
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p * b[i + 1] - q * b[i + 2]

        # Уточнення p та q
        q_new = a[0] / b[2]
        p_new = (a[1] * b[2] - a[0] * b[3]) / (b[2] ** 2)

        # Обчислення коренів alpha та beta
        alpha = -p_new / 2
        # Перевірка підкореневого виразу на від'ємний знак
        det = q_new - alpha ** 2
        beta = np.sqrt(max(0, det))

        # Критерії зупинки
        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            return complex(alpha, beta), complex(alpha, -beta), iter_count

        p, q = p_new, q_new

    return None, None, 1000

# --- 6. Головна програма ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Параметри та точність ---
    eps = 1e-10
    file_trans = "tabulation.csv"
    file_coeffs = "coeffs.txt"

    # --- Крок 1: Табуляція та виділення коренів ---
    print("--- Етап 1: Аналіз трансцендентної функції ---")
    tabulate_and_save(F, -5, 5, 0.1, file_trans)
    x_data, y_data = read_data(file_trans)
    found_roots = find_root_intervals(x_data, y_data)

    # Вибираємо два корені з різною поведінкою
    root_up = next((r for r in found_roots if r["behavior"] == "зростає"), None)
    root_down = next((r for r in found_roots if r["behavior"] == "спадає"), None)

    target_roots = []
    if root_up:
        target_roots.append(root_up)
    if root_down:
        target_roots.append(root_down)

    # --- Крок 2: Уточнення коренів ---
    for root_info in target_roots:
        x0_start = root_info["x0"]
        x_left, x_right = root_info["interval"]
        beh = root_info["behavior"]

        print(f"\nУточнення кореня (функція {beh}) біля x ≈ {x0_start:.4f}:")
        print(f"{'Метод':<25} | {'Результат':<15} | {'Ітерації':<10}")
        print("-" * 55)

        # 1. Проста ітерація (релаксація)
        # Для зростаючої функції tau < 0, для спадної tau > 0
        tau = -1.0 if beh == "зростає" else 1.0
        res, it = simple_iteration(F, x0_start, tau, eps)
        print(f"{'Проста ітерація':<25} | {res:<15.10f} | {it:<10}")

        # 2. Метод Ньютона
        res, it = newton_method(F, Fp, x0_start, eps)
        print(f"{'Метод Ньютона':<25} | {res:<15.10f} | {it:<10}")

        # 3. Метод Чебишева
        res, it = chebyshev_method(F, Fp, F2p, x0_start, eps)
        print(f"{'Метод Чебишева':<25} | {res:<15.10f} | {it:<10}")

        # 4. Метод хорд
        res, it = chord_method(F, x_left, x_right, eps)
        print(f"{'Метод хорд':<25} | {res:<15.10f} | {it:<10}")

        # 5. Метод парабол
        # Беремо три точки: межі інтервалу та середину
        res, it = parabola_method(F, x_left, x0_start, x_right, eps)
        print(f"{'Метод парабол':<25} | {res:<15.10f} | {it:<10}")

        # 6. Зворотна інтерполяція
        res, it = inverse_interpolation(F, x_left, x0_start, x_right, eps)
        print(f"{'Зворотна інтерполяція':<25} | {res:<15.10f} | {it:<10}")

    # --- Крок 3: Алгебраїчне рівняння ---
    print("\n--- Етап 2: Алгебраїчне рівняння 3-го порядку ---")
    # Коефіцієнти для x^3 - x^2 + x - 1 = 0 (корені: 1, i, -i)
    my_coeffs = np.array([-1.0, 1.0, -1.0, 1.0])
    save_coefficients(my_coeffs, file_coeffs)
    coeffs_from_file = read_coefficients(file_coeffs)

    # 1. Пошук дійсного кореня (Горнер-Ньютон)
    # Початкове наближення 0.5 (близько до очікуваного кореня 1)
    real_root, it_h = horner_newton(coeffs_from_file, 0.5, eps)
    print(f"Дійсний корінь (Горнер-Ньютон): {real_root:.10f} (Ітерацій: {it_h})")

    # 2. Пошук комплексних коренів (Метод Ліна)
    # Шукаємо квадратичний дільник p=0.1, q=0.5
    c1, c2, it_l = lin_method(coeffs_from_file, 0.0, 0.0, eps)
    print(f"Комплексні корені (Метод Ліна):")
    print(f"  z1 = {c1}")
    print(f"  z2 = {c2}")
    print(f"  Кількість ітерацій: {it_l}")

    # Візуалізація
    x_plot = np.linspace(-1, 2, 200)
    y_plot = [poly_eval(coeffs_from_file, x) for x in x_plot]

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, label="F(x) = x^3 - x^2 + x - 1", linewidth=2)

    # Додаємо осі
    plt.axhline(0, color='black', lw=1.5)
    plt.axvline(0, color='black', lw=1)

    plt.ylim(-5, 5)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.title("Графік алгебраїчного рівняння")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.show()