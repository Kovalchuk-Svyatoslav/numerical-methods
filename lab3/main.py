import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------------
# 1. Вхідні дані
# -------------------------------

def read_data(filename):
    x = []
    y =[]
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return x,y

# -------------------------------
# 2. Функції МНК
# -------------------------------

def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(len(x)):
                A[i, j] += x[k]**(i + j)
    return A

def form_vector(x, y, m):
    b = np.zeros(m + 1)
    for i in range(m + 1):
        for k in range(len(x)):
            b[i] += y[k] * x[k]**i
    return b

def gauss_solve(A, b, m):
    for k in range(m + 1):
        max_element = A[k, k]
        max_row = k
        for j in range(k + 1, m + 1):
            if abs(A[j, k]) > abs(max_element):
                max_element = abs(A[j, k])
                max_row = j
        if max_row != k:
            A[k], A[max_row] = A[max_row].copy(), A[k].copy()
            b[k], b[max_row] = b[max_row], b[k]
        if abs(A[k, k]) < 1e-12:
            continue
        for i in range(k + 1, m + 1):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]
    x_sol = np.zeros(m + 1)
    for i in range(m, -1, -1):
        sum_val = 0
        for j in range(i + 1, m + 1):
            sum_val += A[i, j] * x_sol[j]
        x_sol[i] = (b[i] - sum_val) / A[i, i]
    return x_sol

def polynomial(x, coef):
    x_poly = np.array(x)
    y_poly = np.zeros(len(x_poly))
    for i in range(len(coef)):
        y_poly += coef[i] * x_poly**i
    return y_poly

def variance(y_true, y_approx):
    return np.sum((y_true - y_approx) ** 2) / len(y_true)


# -------------------------------
# 3. Вибір оптимального ступеня полінома
# -------------------------------
if __name__ == '__main__':
    x_nodes, y_nodes = read_data('data.csv')
    print("Місяці: ", x_nodes)
    print("Температури: ", y_nodes)

    y_full = y_nodes.copy()
    max_degree = 1
    variances = []

    for m in range(1, max_degree + 1):
        A = form_matrix(x_nodes, m)
        b_vec = form_vector(x_nodes, y_nodes, m)
        coef = gauss_solve(A, b_vec, m)
        y_approx = polynomial(x_nodes, coef)
        var = variance(y_full, y_approx)
        variances.append(var)

    optimal_m = np.argmin(variances)
    actual_optimal_m = optimal_m + 1

    # -------------------------------
    # 4. Побудова апроксимації
    # -------------------------------

    A = form_matrix(x_nodes, actual_optimal_m)
    b_vec = form_vector(x_nodes, y_nodes, actual_optimal_m)
    coef = gauss_solve(A, b_vec, actual_optimal_m)
    y_approx = polynomial(x_nodes, coef)

    # -------------------------------
    # 5. Прогноз на наступні 3 місяці
    # -------------------------------

    x_future = [25,26,27]
    y_future = polynomial(x_future, coef)

    # -------------------------------
    # 6. Похибка апроксимації
    # -------------------------------

    error = np.abs(y_nodes - y_approx)

    # -------------------------------
    # 7. Вивід результатів
    # -------------------------------

    print("\n--- Результати аналізу дисперсії ---")
    for i, v in enumerate(variances):
        print(f"Ступінь m = {i+1}: Дисперсія = {v:.6f}")

    print(f"\nОбраний оптимальний ступінь: m = {actual_optimal_m}")

    print(f"\nПрогноз температури на наступні 3 місяці:")
    for m_n, temp in zip([25, 26, 27], y_future):
        print(f"Місяць {m_n}: {temp:.2f}°C")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(x_nodes, y_nodes, color='blue', label='Реальні дані')
    plt.plot(x_nodes, y_approx, color='red', label=f'МНК (m={actual_optimal_m})')
    plt.title('Апроксимація температури')
    plt.legend()
    plt.grid(True, ls=':')

    plt.subplot(2, 1, 2)
    plt.bar(x_nodes, error, color='gray', label='Похибка')
    plt.axhline(y=np.mean(error), color='green', ls='--', label='Сер. похибка')
    plt.title('Графік похибок')
    plt.legend()
    plt.tight_layout()
    plt.show()