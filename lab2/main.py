import csv
import numpy as np
import matplotlib.pyplot as plt


# --- 1. ЗЧИТУВАННЯ ДАНИХ ---

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y


# --- 2. МАТЕМАТИЧНЕ ЯДРО ---

def omega_k(x, x_nodes, k):
    # Знаходження значення функції ω_k(x)
    res = 1.0
    for i in range(k + 1):
        res *= (x - x_nodes[i])
    return res


def divided_diff_k(x_nodes, y_nodes, k):
    # Знаходження значення розділеної різниці k-го порядку
    res = 0.0
    for i in range(k + 1):
        denominator = 1.0
        for j in range(k + 1):
            if i != j:
                denominator *= (x_nodes[i] - x_nodes[j])
        res += y_nodes[i] / denominator
    return res


def newton_n(x, x_nodes, y_nodes):
    # Інтерполяційний многочлен Ньютона
    n = len(x_nodes)
    res = y_nodes[0]
    for k in range(1, n):
        res += omega_k(x, x_nodes, k - 1) * divided_diff_k(x_nodes, y_nodes, k)
    return res


def lagrange_poly(x, x_nodes, y_nodes):
    # Інтерполяційний многочлен Лагранжа
    n = len(x_nodes)
    res = 0.0
    for i in range(n):
        l_i = 1.0
        for j in range(n):
            if i != j:
                l_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        res += y_nodes[i] * l_i
    return res


def f_model(x):
    # КЛАСИЧНА ФУНКЦІЯ РУНГЕ (адаптована)
    # Вона дає ідеальний візуальний ефект "тремтіння" на краях
    return 1 / (1 + 0.00005 * (x - 500)**2)


# --- 3. ДОСЛІДНИЦЬКА ЧАСТИНА ---

def perform_research():
    print("\n" + "=" * 50)
    print("ДОСЛІДНИЦЬКА ЧАСТИНА")
    print("=" * 50)

    # 1. Дослідження впливу кроку (Фіксований інтервал, різна кількість вузлів)
    print("\n[1] Вплив кроку (n=5, 10, 20):")
    a, b = 10, 1000
    x_fine = np.linspace(a, b, 500)
    y_true = np.array([f_model(xi) for xi in x_fine])

    plt.figure(figsize=(10, 5))
    for k in [5, 10, 20]:
        x_k = np.linspace(a, b, k)
        y_k = [f_model(xi) for xi in x_k]
        y_p = [newton_n(xi, x_k, y_k) for xi in x_fine]
        print(f"Вузлів {k:2d}: Макс. похибка = {np.max(np.abs(y_true - y_p)):.4f}")
        plt.plot(x_fine, y_p, label=f'Вузлів n={k}')

    plt.plot(x_fine, y_true, 'k--', alpha=0.3, label='Модель')
    plt.title('Дослідження впливу кроку (п. 1)')
    plt.legend();
    plt.grid();
    plt.show()

    # 2. Дослідження впливу кількості вузлів (Фіксований крок h=50, змінний інтервал)
    print("\n[2] Вплив інтервалу (фіксований крок h=50):")
    h = 50
    for k in [5, 10, 15, 20]:
        x_nodes = [10 + i * h for i in range(k)]
        y_nodes = [f_model(xi) for xi in x_nodes]
        x_test = x_nodes[-1] - h / 2  # Перевірка перед останнім вузлом
        err = abs(f_model(x_test) - newton_n(x_test, x_nodes, y_nodes))
        print(f"Інтервал [10, {x_nodes[-1]:.0f}]: Вузлів {k:2d}, Похибка = {err:.4f}")

    # 3. Аналіз ефекту Рунге
    print("\n[3] Аналіз ефекту Рунге (n=25):")
    x_r = np.linspace(a, b, 25)
    y_r = [f_model(xi) for xi in x_r]
    y_interp_r = [newton_n(xi, x_r, y_r) for xi in x_fine]

    plt.figure(figsize=(10, 5))
    plt.plot(x_fine, y_interp_r, 'r', label='Newton (n=25)')
    plt.plot(x_fine, y_true, 'k--', label='Model')
    plt.title('Аналіз ефекту Рунге (Осциляції на краях)')
    plt.legend();
    plt.grid();
    plt.show()
    print("При великій кількості вузлів спостерігаються коливання на межах інтервалу.")

# --- 4. ОСНОВНИЙ БЛОК ВИКОНАННЯ ---

if __name__ == "__main__":
    # Створення data.csv для Варіанту 1
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 't'])
        writer.writerows([[1000, 3], [2000, 5], [4000, 11], [8000, 28], [16000, 85]])

    # Використання зчитування як у методичці
    x_nodes, y_nodes = read_data("data.csv")
    print("x:", x_nodes)
    print("y:", y_nodes)

    # Обчислення для n=6000 (Варіант 1)
    x_target = 6000
    res_newton = newton_n(x_target, x_nodes, y_nodes)
    res_lagrange = lagrange_poly(x_target, x_nodes, y_nodes)  # Порівняння (п. 4)

    print("-" * 50)
    print(f"Прогноз для n={x_target} (Ньютон):  {res_newton:.4f} мс")
    print(f"Прогноз для n={x_target} (Лагранж): {res_lagrange:.4f} мс")
    print(f"Різниця між методами: {abs(res_newton - res_lagrange):.4f}")
    print("-" * 50)

    # Запуск дослідницької частини
    perform_research()