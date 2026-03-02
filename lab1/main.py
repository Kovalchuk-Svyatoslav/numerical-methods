import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Запит до Open-Elevation API
# -------------------------------

# Вказуємо URL з координатами точок маршруту
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

# Виконуємо GET-запит до API
response = requests.get(url)
data = response.json() # Перетворюємо відповідь на JSON
results = data["results"] # Витягуємо список результатів

n = len(results) # Кількість точок
print("Кількість вузлів:", n)

# Виводимо табуляцію вузлів
print("\nТабуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
 print(f"{i:2d} | {point['latitude']:.6f} | "
       f"{point['longitude']:.6f} | "
       f"{point['elevation']:.2f}")

# -------------------------------
# 2. Кумулятивна відстань
# -------------------------------

# Функція для обчислення відстані між двома GPS точками
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2) # Переводимо широти у радіани
    dphi = np.radians(lat2 - lat1) # Різниця широт у радіанах
    dlambda = np.radians(lon2 - lon1) # Різниця довгот у радіанах
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a)) # Відстань у метрах

# Створюємо списки координат і висот
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

# Обчислюємо кумулятивну відстань
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

# Виводимо табуляцію відстань/висота
print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

# -------------------------------
# 2b. Запис табуляції у текстовий файл
# -------------------------------

with open("tabulation.txt", "w") as f:
     f.write("№ | Latitude | Longitude | Elevation (m) | Distance (m)\n")
     for i, point in enumerate(results):
         f.write(f"{i:2d} | {point['latitude']:.6f} | "
                 f"{point['longitude']:.6f} | "
                 f"{point['elevation']:.2f} | "
                 f"{distances[i]:.2f}\n")

# -------------------------------
# 3. Кубічний сплайн
# -------------------------------

import numpy as np
def cubic_spline_natural(x, y):
    """
    Обчислення натурального кубічного сплайна для заданих вузлів x і значень y.
    Повертає коефіцієнти a, b, c, d для кожного інтервалу [x_i, x_{i+1}]:
    S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    """

    n = len(x) # Кількість вузлів
    h = np.diff(x) # Відстані між сусідніми вузлами (h[i] = x[i+1] - x[i])

    # Ініціалізація трьохдіагональної системи для знаходження M (других похідних)
    A = np.zeros(n) # піддіагональ
    B = np.zeros(n) # головна діагональ
    C = np.zeros(n) # наддіагональ
    D = np.zeros(n) # праві частини рівнянь

    # Натуральний сплайн: друга похідна в кінцях дорівнює 0
    # Тому на кінцях B[0] = B[n-1] = 1, а D[0] = D[n-1] = 0 (за замовчуванням)
    B[0] = 1
    B[-1] = 1

    # Формування внутрішніх рядків трьохдіагональної системи
    # Це рівняння походять із умови згладженості сплайна (похідні суміжних інтервалів)
    for i in range(1, n-1):
        A[i] = h[i-1] # піддіагональ
        B[i] = 2*(h[i-1] + h[i]) # головна діагональ
        C[i] = h[i] # наддіагональ
        # Різниця нахилів сусідніх відрізків, помножена на 6
        D[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    # Рішення трьохдіагональної системи методом прогонки (forward elimination)
    for i in range(1, n):
        m = A[i]/B[i-1] # коефіцієнт для виключення піддіагоналі
        B[i] -= m*C[i-1] # модифікація головної діагоналі
        D[i] -= m*D[i-1] # модифікація правої частини

    # Зворотний хід (back substitution) для знаходження M
    M = np.zeros(n) # масив других похідних
    M[-1] = D[-1]/B[-1] # останнє значення M

    for i in range(n-2, -1, -1): # рухаємося з кінця до початку
        M[i] = (D[i] - C[i]*M[i+1]) / B[i]

    # Обчислення коефіцієнтів сплайна
    a = y[:-1] # коефіцієнт a = значення функції в лівому кінці інтервалу
    b = np.zeros(n-1) # коефіцієнт b = лінійний член (нахил)
    c = M[:-1]/2 # коефіцієнт c = половина другої похідної
    d = np.zeros(n-1) # коефіцієнт d = кубічний член

    # Обчислення b і d для кожного інтервалу
    for i in range(n-1):
        # Нахил на відрізку з урахуванням кривизни
        b[i] = (y[i+1]-y[i])/h[i] - h[i]*(2*M[i]+M[i+1])/6
        # Кубічний коефіцієнт
        d[i] = (M[i+1]-M[i])/(6*h[i])

    # Повертаємо коефіцієнти для кожного інтервалу і вузли
    return a, b, c, d, x

def spline_eval(xi, a, b, c, d, x_nodes):
    # Обчислення значення сплайна в точці xi
    for i in range(len(x_nodes)-1):
        if x_nodes[i] <= xi <= x_nodes[i+1]:
            dx = xi - x_nodes[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return None

x_full = np.array(distances)
y_full = np.array(elevations)

# Побудова сплайна по всіх вузлах
a_full, b_full, c_full, d_full, x_nodes_full = cubic_spline_natural(x_full, y_full)

xx = np.linspace(x_full[0], x_full[-1], 1000)
yy_full = np.array([spline_eval(xi, a_full, b_full, c_full, d_full, x_nodes_full) for xi in xx])

# -------------------------------
# 4. Аналіз вузлів
# -------------------------------

def test_nodes(k):
    indices = np.linspace(0, len(x_full)-1, k, dtype=int)
    x_k = x_full[indices]
    y_k = y_full[indices]
    a_k, b_k, c_k, d_k, x_nodes_k = cubic_spline_natural(x_k, y_k)
    yy_k = np.array([spline_eval(xi, a_k, b_k, c_k, d_k, x_nodes_k) for xi in xx])
    error = np.abs(yy_k - yy_full)
    print(f"\n===== {k} вузлів =====")
    print("Максимальна похибка:", np.max(error))
    print("Середня похибка:", np.mean(error))
    return yy_k, error

yy_10, err_10 = test_nodes(10)
yy_15, err_15 = test_nodes(15)
yy_20, err_20 = test_nodes(20)

plt.figure()
plt.plot(xx, yy_full, label="21 вузол (еталон)")
plt.plot(xx, yy_10, label="10 вузлів")
plt.plot(xx, yy_15, label="15 вузлів")
plt.plot(xx, yy_20, label="20 вузлів")
plt.legend()
plt.title("Вплив кількості вузлів")
plt.show()

plt.figure()
plt.plot(xx, err_10, label="10 вузлів")
plt.plot(xx, err_15, label="15 вузлів")
plt.plot(xx, err_20, label="20 вузлів")
plt.legend()
plt.title("Похибка апроксимації")
plt.show()

# -------------------------------
# 5. Графік залежності кумулятивної відстані від висоти
# -------------------------------

plt.figure()
plt.plot(distances, elevations, 'o-', color='green')
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Висота маршруту від кумулятивної відстані")
plt.grid(True)
plt.show()

# -------------------------------
# 6. Характеристики маршруту
# -------------------------------

print("\n===== Характеристики маршруту =====")
print("Загальна довжина маршруту (м):", distances[-1])

total_ascent = sum(max(elevations[i]-elevations[i-1],0) for i in range(1,n))
print("Сумарний набір висоти (м):", total_ascent)

total_descent = sum(max(elevations[i-1]-elevations[i],0) for i in range(1,n))
print("Сумарний спуск (м):", total_descent)

# -------------------------------
# 7. Аналіз градієнта
# -------------------------------

grad_full = np.gradient(yy_full, xx) * 100
print("\n===== Аналіз градієнта =====")
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))

steep_sections = np.where(np.abs(grad_full) > 15)[0]
print("Кількість ділянок з крутизною > 15%:", len(steep_sections))

# -------------------------------
# 8. Механічна енергія підйому
# -------------------------------

mass = 80
g = 9.81
energy = mass * g * total_ascent

print("\n===== Механічна робота =====")
print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy/1000)
print("Енергія (ккал):", energy / 4184)