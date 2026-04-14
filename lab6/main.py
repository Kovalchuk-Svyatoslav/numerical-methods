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
def calculate_norm(vector):
    return max(abs(x) for x in vector)

# --- 2. Генерація та зчитування даних ---

# Генерація випадкової матриці A розмірності n x n та запис у файл
def generate_matrix(a, b, n, file_A):
    A = np.random.uniform(a, b, (n, n))
    np.savetxt(file_A, A)

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

# --- 3. LU-розклад ---

# Знаходження LU-розкладу матриці А та запис результату в текстові файли
def lu_decomposition(A, n, file_L, file_U):
    L = np.zeros((n, n))
    U = np.eye(n) # Діагональні елементи U рівні 1
    for k in range(n):
        # Обчислення елементів стовпця L
        for i in range(k, n):
            s_l = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - s_l

        # Перевірка на нульовий діагональний елемент перед діленням
        if abs(L[k][k]) < 1e-15:
            raise ValueError(f"Помилка: Головний елемент L[{k}][{k}] близький до нуля!")

        # Обчислення елементів рядка U
        for i in range(k + 1, n):
            s_u = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - s_u) / L[k][k]

    np.savetxt(file_L, L)
    np.savetxt(file_U, U)

# --- 4. Розв'язання системи ---

# Розв'язок системи AX=B за допомогою LU-розкладу
def solve_lu(L, U, B, n):
    # Прямий хід: LZ = B
    Z = np.zeros(n)
    Z[0] = B[0] / L[0][0]
    for k in range(1, n):
        sum_z = sum(L[k][j] * Z[j] for j in range(k))
        Z[k] = (1 / L[k][k]) * (B[k] - sum_z)

    # Зворотний хід: UX = Z
    X = np.zeros(n)
    X[n - 1] = Z[n - 1]

    for k in range(n - 2, -1, -1):
        sum_x = sum(U[k][j] * X[j] for j in range(k + 1, n))
        X[k] = (Z[k] - sum_x)
    return X

# --- 5. Ітераційне уточнення ---
def refine_solution(A, B, L, U, x_initial, n, eps):
    x_current = x_initial.copy()
    prev_norm = float('inf')

    for iteration in range(100):
        # 1. Вектор нев'язки R = B - AX
        AX = matrix_vector_multiply(A, x_current, n)
        R = [B[i] - AX[i] for i in range(n)]

        # 2. Знаходження вектора похибки delta_x із системи A * delta_x = R
        delta_x = solve_lu(L, U, R, n)

        # 3. Уточнення розв'язку X = X + delta_x
        x_current = [x_current[i] + delta_x[i] for i in range(n)]

        current_norm = calculate_norm(delta_x)

        # Перевірка умови зупинки за нормою нев'язки
        if current_norm <= eps:
            return x_current, iteration

        if current_norm >= prev_norm and iteration > 10:
            print(f"Попередження: Досягнуто межі точності на {iteration} ітерації.")
            return x_current, iteration

        prev_norm = current_norm

    print("Метод не зійшовся за відведену кількість ітерацій")
    return x_current, 100

# --- ГОЛОВНА ПРОГРАМА ---

if __name__ == "__main__":
    n = 100
    eps0 = 1e-14

    file_A = "matrix_A.txt"
    file_B = "vector_B.txt"
    file_L = "matrix_L.txt"
    file_U = "matrix_U.txt"

    # --- Генерація та зчитування даних ---
    generate_matrix(-10, 11, n, file_A)
    A = read_matrix(file_A)

    generate_vector_b(A, 2.5, n, file_B)
    B = read_vector(file_B)

    # --- LU-розклад ---
    lu_decomposition(A, n, file_L, file_U)
    L = read_matrix(file_L)
    U = read_matrix(file_U)

    # --- Початковий розв'язок та похибка ---
    x0 = solve_lu(L, U, B, n)

    # Оцінка точності знайденого розв'язку
    R0 = [B[i] - matrix_vector_multiply(A, x0, n)[i] for i in range(n)]
    initial_err = calculate_norm(R0)
    print(f"Початкова похибка нев'язки (eps): {initial_err:.2e}")

    # --- Ітераційне уточнення розв'язку ---
    x_final, iters = refine_solution(A, B, L, U, x0, n, eps0)

    # Фінальна перевірка
    RF = [B[i] - matrix_vector_multiply(A, x_final, n)[i] for i in range(n)]
    final_err = calculate_norm(RF)

    print("-" * 40)
    print(f"Уточнений розв'язок знайдено за {iters} ітерацій.")
    print(f"Фінальна похибка нев'язки: {final_err:.2e}")
    clean_output = [round(float(val), 14) for val in x_final[:5]]
    print(f"5 перших елементів розв'язку: {clean_output}")