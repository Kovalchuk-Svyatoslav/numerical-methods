import numpy as np
import matplotlib.pyplot as plt

# --- 1. МАТЕМАТИЧНА МОДЕЛЬ---

#Модель вологості ґрунту
def m(t):
    return 50 * np.exp(-0.1*t) + 5 * np.sin(t)

# Точна похідна (аналітичний розв'язок)
def m_exact_derivative(t):
    return -5 * np.exp(-0.1*t) + 5 * np.cos(t)

# --- 2. МЕТОДИ ЧИСЕЛЬНОГО ДИФЕРЕНЦІЮВАННЯ ---

# Апроксимація за центральною різницею (базовий метод)
def central_diff(t, h):
    return (m(t + h) - m(t - h)) / (2 * h)

# Метод Рунге-Ромберга для уточнення
def runge_romberg(y_h, y_2h):
    return y_h + (y_h - y_2h) / 3

# Метод Ейткена: уточнене значення та порядок точності p
def aitken(y_h, y_2h, y_4h):
    # Уточнене значення
    y_refined =(y_2h**2 - y_4h * y_h) / (2 * y_2h - (y_4h + y_h))

    # Оцінка фактичного порядку точності p
    p_order = np.log(abs((y_4h - y_2h) / (y_2h - y_h))) / np.log(2)
    return y_refined, p_order

# --- 3. ФУНКЦІЯ ДЛЯ ОБЧИСЛЕННЯ ПОХИБКИ ---

def get_err(approx_val, exact_val):
    return abs(approx_val - exact_val)

# --- 4. ГОЛОВНИЙ БЛОК ВИКОНАННЯ ---

if __name__ == "__main__":
    # Обчислення точного значення похідної
    t0 = 1
    exact_val = m_exact_derivative(t0)
    print(f"--- Аналітика ---")
    print(f"Точне значення похідної: {exact_val:.8f}\n")

    # Пошук оптимального кроку h0
    h_range = 10.0**np.arange(-20,4)
    errors = [get_err(central_diff(t0, h), exact_val) for h in h_range]
    best_idx = np.argmin(errors)

    h0 = h_range[best_idx]
    R0 = errors[best_idx]
    print(f"--- Оптимізація ---")
    print(f"Оптимальний крок h0: {h0:.0e}")
    print(f"Найкраща точність R0: {R0:.2e}")


    h = 10**-3

    # Базова чисельна похідна та R1
    y_h = central_diff(t0, h)
    R1 = get_err(y_h, exact_val)

    # Рунге-Ромберг та R2
    y_2h = central_diff(t0, 2*h)
    y_R = runge_romberg(y_h, y_2h)
    R2 = get_err(y_R, exact_val)

    # Ейткен та R3
    y_4h = central_diff(t0, 4*h)
    y_E, p_val = aitken(y_h, y_2h, y_4h)
    R3 = get_err(y_E, exact_val)

    # Вивід результатів
    print(f"--- Уточнення результатів (h={h}) ---")
    print(f"{'Метод':<25} | {'Значення':<12} | {'Похибка R':<10}")
    print("-" * 55)
    print(f"{'Центральна різниця (R1)':<25} | {y_h:<12.7f} | {R1:.2e}")
    print(f"{'Рунге-Ромберг (R2)':<25} | {y_R:<12.7f} | {R2:.2e}")
    print(f"{'Метод Ейткена (R3)':<25} | {y_E:<12.7f} | {R3:.2e}")
    print("-" * 55)
    print(f"Оцінений порядок точності p: {p_val:.4f}")

    t_plot = np.linspace(0, 20, 400)  # Проміжок часу від 0 до 20
    plt.figure(figsize=(8, 5))
    plt.plot(t_plot, m(t_plot), label='M(t) = 50e^{-0.1t} + 5sin(t)')
    plt.title('Модель вологості ґрунту')
    plt.xlabel('Час t')
    plt.ylabel('Вологість M(t)')
    plt.grid(True, ls=':')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.loglog(h_range, errors, 'b-o', label='Похибка R(h)')
    plt.axvline(x=h0, color='r', linestyle='--', label=f'Оптимальний h0={h0:.0e}')

    plt.title("Залежність похибки від кроку диференціювання")
    plt.xlabel("Крок h (log scale)")
    plt.ylabel("Абсолютна похибка R (log scale)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()