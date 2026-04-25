import numpy as np

# --- 1. Допоміжні функції для обчислень ---

# Обчислення добутку матриці на вектор
def matrix_vector_multiply(A, X, n):
    B = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * X[j]
        B[i] = s
    return B

# Обчислення норми вектора
def vector_norm(vector):
    return max(abs(x) for x in vector)

# Обчислення норми матриці
def matrix_norm(matrix):
    # Створення списку сум модулів для кожного рядка
    row_sums = []
    for row in matrix:
        s = sum(abs(x) for x in row)
        row_sums.append(s)
    # Повертання найбільшої з цих сум
    return max(row_sums)

# --- 2. Генерація та зчитування даних ---

# Генерація випадкової матриці A розмірності n x n та запис у файл
def generate_matrix(a, b, n, file_A):
    A = np.random.uniform(a, b, (n, n))
    # Діагональне переважання
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 5)
    np.savetxt(file_A, A)
    return A

# Обчислення вектора B на основі заданого розв'язку X та запис у файл
def generate_vector_b(A, x_value, n, file_B):
    x_exact = np.full(n, x_value)
    B = matrix_vector_multiply(A, x_exact, n)
    np.savetxt(file_B, B)

# Зчитування матриці А з текстового файлу
def read_matrix(file):
    matrix = np.loadtxt(file)
    return matrix

# Зчитування вектора В з текстового файлу
def read_vector(file_B):
    vector = np.loadtxt(file_B)
    return vector

# --- 3. Методи розв'язку СЛАР ---

# Метод простої ітерації
def simple_iteration_method(A, B, x, n, eps):
    tau = 1.0 / matrix_norm(A)

    # 1. Формування матриці C та вектора d
    E = np.eye(n)
    C = E - tau * A
    d = tau * B

    # 2. Перевірка умови збіжності
    if matrix_norm(C) >= 1:
        print("Попередження: Умова збіжності ||C|| < 1 не виконується!")

    x_curr = x.copy()

    for iteration in range(1, 10001):
        # Обчислення наступного наближення: x_next = C * x_curr + d
        x_next = matrix_vector_multiply(C, x_curr, n) + d

        # Перевірка умови закінчення за нормою різниці
        if vector_norm(x_next - x_curr) < eps:
            return x_next, iteration

        x_curr = x_next.copy()

    return x_curr, 10000

# Метод Якобі
def jacobi_method(A, B, x, n, eps):
    x_curr = x.copy()

    for iteration in range(1, 10001):
        x_next = np.zeros(n)

        for i in range(n):
            # Сума a_ij * x_j (де j != i)
            s = 0.0
            for j in range(n):
                if i != j:
                    s += A[i][j] * x_curr[j]

            # Формула Якобі
            if abs(A[i][i]) < 1e-15:
                raise ValueError(f"Діагональний елемент A[{i}][{i}] занадто малий!")

            x_next[i] = (B[i] - s) / A[i][i]

        # Критерій закінчення: ||x_next - x_curr|| < eps
        if vector_norm(x_next - x_curr) < eps:
            return x_next, iteration

        x_curr = x_next.copy()

    return x_curr, 10000

# Метод Зейделя
def seidel_method(A, B, x, n, eps):
    x_curr = x.copy()

    for iteration in range(1, 10001):
        x_prev = x_curr.copy()

        for i in range(n):
            # Сума для елементів, які вже оновлені
            s1 = sum(A[i][j] * x_curr[j] for j in range(i))

            # Сума для елементів, які ще старі
            s2 = sum(A[i][j] * x_curr[j] for j in range(i + 1, n))

            # Формула Зейделя
            if abs(A[i][i]) < 1e-15:
                raise ValueError(f"Діагональний елемент A[{i}][{i}] занадто малий!")

            x_curr[i] = (B[i] - s1 - s2) / A[i][i]

        # Критерій закінчення: ||x_new - x_old|| < eps
        if vector_norm(x_curr - x_prev) < eps:
            return x_curr, iteration

    return x_curr, 10000

if __name__ == "__main__":
    # Параметри лабораторної роботи
    N = 100
    eps0 = 1e-14
    x_target = 2.5

    file_A = "matrix_A.txt"
    file_B = "vector_B.txt"

    # 1. Підготовка даних
    generate_matrix(-10, 10, N, file_A)
    A = read_matrix(file_A)
    generate_vector_b(A, x_target, N, file_B)
    B = read_vector(file_B)

    # 2. Початкове наближення
    x0 = np.ones(N)

    print(f"Цільова точність обчислення СЛАР розмірністю {N}x{N}: {eps0}")
    print("-" * 60)

    # 3. Розв'язання різними методами
    # Метод простої ітерації
    x_si, iters_si = simple_iteration_method(A, B, x0, N, eps0)

    # Метод Якобі
    x_jac, iters_jac = jacobi_method(A, B, x0, N, eps0)

    # Метод Зейделя
    x_sei, iters_sei = seidel_method(A, B, x0, N, eps0)

    # 4. Вивід результатів порівняння
    print(f"{'Метод':<25} | {'Ітерацій':<10} | {'Перші 3 елементи X'}")
    print("-" * 60)

    results = [
        ("Простої ітерації", iters_si, x_si),
        ("Якобі", iters_jac, x_jac),
        ("Зейделя", iters_sei, x_sei)
    ]

    for name, iters, sol in results:
        clean_sol = [round(float(val), 10) for val in sol[:3]]
        print(f"{name:<25} | {iters:<10} | {clean_sol}")