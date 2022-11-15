"""
Обучаем нейрон распознавать дорожные знаки.
Для данной задачи создал изображения в Paint размером 100 на 100 пикселей:
- знак ограничения 20 км
- знак ограничения 40 км
- знак ограничения 60 км
- знак ограничения 100 км

Для тестирования сделал следующие изображения:
- знак с 0
- знак ограничения 21 км
- знак ограничения 40 км, перечеркнутый
- знак ограничения 50 км
- знак ограничения 50 км, почти полностью стертый
- знак ограничения 100 км, с качеством изображения в 2 раза меньше (50 на 50 пикселей)
- знак без цифры (только круг)
- знак без цифры (только круг) с диагональной чертой

Каждую тестовую картинку сравню с каждым знаком ограничения скорости и выведу все в словарь.
"""
import numpy as np
import cv2 as cv


# Считываем изображения с дорожными знаками:
img_1 = "images/20.png"
img_2 = "images/40.png"
img_3 = "images/60.png"
img_4 = "images/100.png"


# Изображения для тестирования обученного нейрона:
test_img_1 = "images/test_img/0.png"
test_img_2 = "images/test_img/21.png"
test_img_3 = "images/test_img/40_line.png"
test_img_4 = "images/test_img/50.png"
test_img_5 = "images/test_img/50cut.png"
test_img_6 = "images/test_img/100min.png"  # размер изображения 50х50 пикселей
test_img_7 = "images/test_img/fon.png"
test_img_8 = "images/test_img/stop.png"


def img_to_arr(img):
    """Перевод изображения в массив"""
    array = cv.imread(img, 0).astype(float)
    return array


# Массивы для обучения нейрона:
learn_arrays = np.array([
    img_to_arr(img_1).flatten(),
    img_to_arr(img_2).flatten(),
    img_to_arr(img_3).flatten(),
    img_to_arr(img_4).flatten(),
])


# Изображение test_img_6 сделаем 100х100 пикселей (увеличиваем масштаб в 2 раза) с помощью метода numpy - kron.
# Так мы пробуем отличить дорожные знаки, которые отдалены в 2 раза:
test_array_6 = img_to_arr(test_img_6)
test_array_6 = np.kron(test_array_6, np.ones((2, 2)))


# Массивы для тестирования нейрона:
test_arrays = np.array([
    img_to_arr(test_img_1).flatten(),
    img_to_arr(test_img_2).flatten(),
    img_to_arr(test_img_3).flatten(),
    img_to_arr(test_img_4).flatten(),
    img_to_arr(test_img_5).flatten(),
    test_array_6.flatten(),
    img_to_arr(test_img_7).flatten(),
    img_to_arr(test_img_8).flatten(),
])


def rgb_to_binary(array):
    """Перевод из rgb (0...255) в бинарные данные (0 или 1)"""
    for i in range(array.shape[0]):
        if array[i] < 50:
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
w = np.zeros((4, 10000)).astype(float)
output_value = np.array([1, 1, 1, 1])
alfa = 0.01
betta = -4
sigma = lambda x: 1 if x > 0 else 0


def f(x, iter):
    s = betta + np.sum(x @ w[iter])
    return sigma(s)


def f_test(x):
    keys = ['20km', '40km', '60km', '100km']
    values = []
    values_dict = {'20km': None,
                   '40km': None,
                   '60km': None,
                   '100km': None}
    for j in range(4):
        s = betta + np.sum(x @ w[j])
        values.append(sigma(s))
    for p in range(4):
        values_dict.update({keys[p]: values[p]})
    return values_dict


i = 0


def train():
    global w, i
    w_copy = w.copy()
    i = 0
    for x, y in zip(learn_arrays, output_value):
        w[i] += alfa * (y - f(x, i)) * x
        i += 1
    return (w != w_copy).any()


while train():
    print('The program is completed.')


# Тестируем нейрон на распознавание:
print(f'Значение 0: {f_test(test_arrays[0])}')
print(f'Значение 21: {f_test(test_arrays[1])}')
print(f'Значение 40 (перечеркнутое): {f_test(test_arrays[2])}')
print(f'Значение 50: {f_test(test_arrays[3])}')
print(f'Значение 50 (частично стерто): {f_test(test_arrays[4])}')
print(f'Значение 100 (смасштабированное в 2 раза): {f_test(test_arrays[5])}')
print(f'Значение пустое (только круг): {f_test(test_arrays[6])}')
print(f'Значение stop: {f_test(test_arrays[7])}')


"""
INPUT:

Примечание: значения в словаре означают совпадает ли или нет (0 - не совпадает, 1 - совпадает)

Значение 0: {'20km': 0, '40km': 0, '60km': 0, '100km': 0}
Значение 21: {'20km': 1, '40km': 0, '60km': 0, '100km': 0}
Значение 40 (перечеркнутое): {'20km': 1, '40km': 1, '60km': 1, '100km': 0}
Значение 50: {'20km': 1, '40km': 1, '60km': 1, '100km': 0}
Значение 50 (частично стерто): {'20km': 0, '40km': 0, '60km': 0, '100km': 0}
Значение 100 (смасштабированное в 2 раза): {'20km': 0, '40km': 0, '60km': 0, '100km': 1}
Значение пустое (только круг): {'20km': 0, '40km': 0, '60km': 0, '100km': 0}
Значение stop: {'20km': 0, '40km': 0, '60km': 0, '100km': 0}

Вывод: большинство картинок нейрон правильно распознает, но не все, 
к примеру 21 распознает как 20, перечеркнутое 40 видит как 20, 40 и 60.
"""
