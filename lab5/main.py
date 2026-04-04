import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- 1. ПІДІНТЕГРАЛЬНА ФУНКЦІЯ (НАВАНТАЖЕННЯ НА СЕРВЕР) ---
def f(x):
    return 50 + 20*np.sin(np.pi*x/12) + 5*np.exp(-0.2*(x - 12)**2)


# --- 2. МЕТОДИ ЧИСЕЛЬНОГО ІНТЕГРУВАННЯ ---
def simpson_method(f_func, a, b, N):
    if N % 2 != 0:
        print("Кількість відрізків N має бути парною!")
        exit()

    h = (b - a) / N # Крок інтегрування
    x = np.linspace(a, b, N + 1) # Вузли x_i
    f_range = f_func(x) # Значення функції у вузлах f_i

    s = f_range[0] + f_range[-1] # Крайні значення f0 та fN
    s += 4 * np.sum(f_range[1:-1:2]) # Вузли з непарними індексами
    s += 2 * np.sum(f_range[2:-1:2]) # Внутрішні вузли з парними індексами
    return  (h/3) * s

def adaptive_simpson(f_func, a, b, eps, current_depth=0):
    h = b - a
    mid = (a + b) / 2

    # Обчислення інтеграла на всьому відрізку (одна парабола)
    I1 = (h / 6) * (f_func(a) + 4 * f_func(mid) + f_func(b))

    # Обчислення інтеграла на двох половинках
    m1 = (a + mid) / 2
    m2 = (mid + b) / 2
    I2 = (h / 12) * (f_func(a) + 4 * f_func(m1) + f_func(mid)) + \
         (h / 12) * (f_func(mid) + 4 * f_func(m2) + f_func(b))

    # Перевірка умови збіжності
    if abs(I1 - I2) <= 15 * eps:  # 15 - коефіцієнт для методу Сімпсона
        return I2 + (I2 - I1) / 15
    else:
        # Рекурсивне ділення відрізків
        return adaptive_simpson(f_func, a, mid, eps / 2, current_depth + 1) + \
            adaptive_simpson(f_func, mid, b, eps / 2, current_depth + 1)


# --- 3. МЕТОДИ ПІДВИЩЕННЯ ТОЧНОСТІ (УТОЧНЕННЯ) ---

def runge_romberg(I_h, I_qh, p=4):
    return I_h + (I_h - I_qh) / (2**p - 1)


def aitken(I1, I2, I3):
    # Уточнене значення інтеграла
    I_refined = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))

    # Розрахунок фактичного порядку точності p
    p_actual = (1 / np.log(2)) * np.log(abs((I3 - I2) / (I2 - I1)))

    return I_refined, p_actual

# --- 4. ГОЛОВНИЙ БЛОК ОБЧИСЛЕНЬ ---

if __name__ == "__main__":
    a, b = 0, 24 # Інтервал часу (доба)
    eps_target = 1e-12 # Задана точність для дослідження

    # Знаходження точного значення I0 через SciPy
    I0, _ = quad(f, a, b)
    print(f"--- ЕТАП 1: ТОЧНЕ ЗНАЧЕННЯ ---\nI0 (еталон): {I0:.14f}\n")

    # Дослідження залежності точності від N
    N_values = np.arange(10, 1001, 2)
    errors = []
    N_opt = None

    for n in N_values:
        I_n = simpson_method(f, a, b, n)
        error = abs(I_n - I0)
        errors.append(error)
        if (N_opt is None) and (error < eps_target):
            N_opt = n

    if N_opt is None: N_opt = 1000
    eps_opt = abs(simpson_method(f, a, b, N_opt) - I0)
    print(f"--- ЕТАП 2: ДОСЛІДЖЕННЯ N ---")
    print(f"Оптимальне N_opt: {N_opt}")
    print(f"Точність при N_opt: {eps_opt:.2e}\n")

    # Обчислення похибки при N0 кратному 8
    N0 = (N_opt // 10)
    N0 = (N0 // 8) * 8
    if N0 < 8: N0 = 8

    I_N0 = simpson_method(f, a, b, N0)
    eps0 = abs(I_N0 - I0)
    print(f"--- ЕТАП 3: БАЗОВА ПОХИБКА (N0={N0}) ---\neps0: {eps0:.2e}\n")

    # Уточнення за Рунге-Ромбергом

    I_half = simpson_method(f, a, b, N0 // 2)
    I_R = runge_romberg(I_N0, I_half)
    epsR = abs(I_R - I0)
    print(f"--- ЕТАП 4: МЕТОД РУНГЕ-РОМБЕРГА ---\nУточнене I_RR: {I_R:.12f}")
    print(f"Похибка epsR: {epsR:.2e}\n")

    # Уточнення за Ейткеном та порядок p
    I_q = simpson_method(f, a, b, N0 // 4)
    I_Aitken, p_val = aitken(I_N0, I_half, I_q)
    epsA = abs(I_Aitken - I0)
    print(f"--- ЕТАП 5: МЕТОД ЕЙТКЕНА ---")
    print(f"Уточнене I_E: {I_Aitken:.12f}")
    print(f"Фактичний порядок точності p: {p_val:.4f}")
    print(f"Похибка epsE: {epsA:.2e}\n")

    # Адаптивний алгоритм
    I_adapt = adaptive_simpson(f, a, b, 1e-7)
    print(f"--- ЕТАП 6: АДАПТИВНИЙ АЛГОРИТМ ---\nРезультат: {I_adapt:.12f}")
    print(f"Похибка: {abs(I_adapt - I0):.2e}\n")

    # --- 5. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ---

    # Побудова графіка функції навантаження
    x_plot = np.linspace(0, 24, 1000)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label=r'$f(x)=50+20\sin(\frac{\pi x}{12})+5e^{-0.2(x-12)^2}$')
    plt.title('Графік інтенсивності навантаження на сервер')
    plt.xlabel('Час, х (год)')
    plt.ylabel('Навантаження, f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, errors, 'b-', label='Похибка Сімпсона |I(N) - I0|')
    plt.axhline(y=eps_target, color='r', linestyle='--', label=f'Ціль (1e-12)')
    plt.scatter([N_opt], [eps_opt], color='green', label=f'N_opt={N_opt}')

    plt.title('Залежність похибки від кількості кроків розбиття N')
    plt.xlabel('N (кількість відрізків)')
    plt.ylabel('Похибка (логарифмічна шкала)')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()