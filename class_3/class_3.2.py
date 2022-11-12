import numpy as np
import cv2 as cv

# Изображения для обучения нейрона:
img_1 = 'images/m_1.png'  # буква М
img_2 = 'images/m_2.png'  # буква М
img_3 = 'images/m_3.png'  # буква М

# Изображения для тестирования обученного нейрона:
img_t1 = 'images/test_1.png'  # буква М
img_t2 = 'images/test_2.png'  # буква М
img_t3 = 'images/test_3.png'  # 2 горизонтальные линии
img_t4 = 'images/test_4.png'  # галочка
img_t5 = 'images/test_5.png'  # буква П
img_t6 = 'images/test_6.png'  # квадрат
img_t7 = 'images/test_7.png'  # круг
img_t8 = 'images/test_8.png'  # буква М (от руки)
img_t9 = 'images/test_9.png'  # буква М (от руки)


def img_to_arr(img):
    """Перевод изображения в массив"""
    array = cv.imread(img, 0).astype(float)
    return array


# Массивы для обучения нейрона:
learn_arrays = np.array([
    img_to_arr(img_1).flatten(),
    img_to_arr(img_2).flatten(),
    img_to_arr(img_3).flatten(),
])

# Массивы для тестирования нейрона:
test_arrays = np.array([
    img_to_arr(img_t1).flatten(),
    img_to_arr(img_t2).flatten(),
    img_to_arr(img_t3).flatten(),
    img_to_arr(img_t4).flatten(),
    img_to_arr(img_t5).flatten(),
    img_to_arr(img_t6).flatten(),
    img_to_arr(img_t7).flatten(),
    img_to_arr(img_t8).flatten(),
    img_to_arr(img_t9).flatten(),
])


def rgb_to_binary(array):
    """Перевод из rgb (0...255) в бинарные данные (0 или 1)"""
    for i in range(array.shape[0]):
        if array[i] == 0:
            array[i] = 1
        else:
            array[i] = 0
    return array


# Переводим массивы для обучения в бинарные данные:
for arr in range(learn_arrays.shape[0]):
    learn_arrays[arr] = rgb_to_binary(learn_arrays[arr])

# Переводим массивы для тестирования в бинарные данные:
for arr in range(test_arrays.shape[0]):
    test_arrays[arr] = rgb_to_binary(test_arrays[arr])

# ------------------------------------------------------------------

# Обучаем нейрон:
w = np.zeros(2500).astype(float)
output_value = np.array([1, 1, 1])
alfa = 0.1
betta = -5
sigma = lambda x: 1 if x > 0 else 0


def f(x):
    s = betta + np.sum(x @ w)
    return sigma(s)


def train():
    global w
    w_copy = w.copy()
    for x, y in zip(learn_arrays, output_value):
        w += alfa * (y - f(x)) * x
    return (w != w_copy).any()


while train():
    pass

# Тестируем нейрон на распознавание:
print(f'Обучающее значение 1: {f(learn_arrays[0])}')
print(f'Обучающее значение 2: {f(learn_arrays[1])}')
print(f'Обучающее значение 3: {f(learn_arrays[2])}')
print(f'Тестовое значение 1 (буква М): {f(test_arrays[0])}')
print(f'Тестовое значение 2 (буква М): {f(test_arrays[1])}')
print(f'Тестовое значение 3 (2 горизонтальные линии): {f(test_arrays[2])}')
print(f'Тестовое значение 4 (галочка): {f(test_arrays[3])}')
print(f'Тестовое значение 5 (буква П): {f(test_arrays[4])}')
print(f'Тестовое значение 6 (квадрат): {f(test_arrays[5])}')
print(f'Тестовое значение 7 (круг): {f(test_arrays[6])}')
print(f'Тестовое значение 8 (буква М (от руки): {f(test_arrays[7])}')
print(f'Тестовое значение 9 (буква М (от руки): {f(test_arrays[8])}')

"""
OUTPUT PROGRAM:
The program is completed.
Обучающее значение 1: 1
Обучающее значение 2: 1
Обучающее значение 3: 1
Тестовое значение 1 (буква М): 1
Тестовое значение 2 (буква М): 1
Тестовое значение 3 (2 горизонтальные линии): 0
Тестовое значение 4 (галочка): 0
Тестовое значение 5 (буква П): 0
Тестовое значение 6 (квадрат): 0
Тестовое значение 7 (круг): 0
Тестовое значение 8 (буква М (от руки): 1
Тестовое значение 9 (буква М (от руки): 1
"""

# ------------------------------------------------------------------

# Оценим чувствительность к повороту изображения:
# Создадим несколько картинок с буквой М с разным углом поворота (5, 10, 20, 40, 90, 180):
img_rt1 = 'images/rotate_5.png'  # 5 градусов
img_rt2 = 'images/rotate_10_1.png'  # 10 градусов
img_rt3 = 'images/rotate_10_2.png'  # 10 градусов
img_rt4 = 'images/rotate_20.png'  # 20 градусов
img_rt5 = 'images/rotate_40.png'  # 40 градусов
img_rt6 = 'images/rotate_90.png'  # 90 градусов
img_rt7 = 'images/rotate_180.png'  # 180 градусов

rotate_arrays = np.array([
    img_to_arr(img_rt1).flatten(),
    img_to_arr(img_rt2).flatten(),
    img_to_arr(img_rt3).flatten(),
    img_to_arr(img_rt4).flatten(),
    img_to_arr(img_rt5).flatten(),
    img_to_arr(img_rt6).flatten(),
    img_to_arr(img_rt7).flatten(),
])

for arr in range(rotate_arrays.shape[0]):
    rotate_arrays[arr] = rgb_to_binary(rotate_arrays[arr])

print(f'Поворот буквы М на 5 градусов: {f(rotate_arrays[0])}')
print(f'Поворот буквы М на 10 градусов: {f(rotate_arrays[1])}')
print(f'Поворот буквы М на 10 градусов: {f(rotate_arrays[2])}')
print(f'Поворот буквы М на 20 градусов: {f(rotate_arrays[3])}')
print(f'Поворот буквы М на 40 градусов: {f(rotate_arrays[4])}')
print(f'Поворот буквы М на 90 градусов: {f(rotate_arrays[5])}')
print(f'Поворот буквы М на 180 градусов: {f(rotate_arrays[6])}')

"""
OUTPUT PROGRAM:
Поворот буквы М на 10 градусов: 0
Поворот буквы М на 20 градусов: 0
Поворот буквы М на 40 градусов: 0
Поворот буквы М на 90 градусов: 0
Поворот буквы М на 180 градусов: 1

ВЫВОД:
Данный нейрон очень чувствителен к поворотам изображения.
В качестве дополнительной проверки, составил цикл с условием, 
что нейрон распознает букву М при минимальном угле поворота, а именно 5 градусов, не забывая про прошлые проверки:

condition = f(test_arrays[0]) == 1 and \
            f(test_arrays[1]) == 1 and \
            f(test_arrays[2]) == 0 and \
            f(test_arrays[3]) == 0 and \
            f(test_arrays[4]) == 0 and \
            f(test_arrays[5]) == 0 and \
            f(test_arrays[6]) == 0 and \
            f(test_arrays[7]) == 1 and \
            f(test_arrays[8]) == 1 and \
            f(rotate_arrays[0]) == 1 and \
            f(rotate_arrays[1]) == 1
            
В итоге, прогнав по циклу данное условие,
while betta > -25:
    betta -= 0.1
    if condition:
        print(betta)

print ничего не выдал.
"""
