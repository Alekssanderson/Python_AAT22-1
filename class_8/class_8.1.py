# Конвертер изображений из .jpg в .png

import glob as gb
import cv2 as cv
import os
from pathlib import Path

s = []
filename = []
images_jpg = gb.glob('Images_JPG/*.jpg')  # находим и сохраняем все файлы .jpg в список из директории Images_JPG/
for image in images_jpg:
    s.append(cv.imread(image))  # переводим изображение в массив и сохраняем в список s
    filename.append(Path(image).stem)  # с помощью Path модуля pathlib извлекаем название файла и сохраняем в список

# Создаем директорию Images_PNG для сохранения конвертированных изображений
# Если такая директория уже существует, то не создаем (иначе ошибка)
dir = 'Images_PNG'
if not os.path.isdir(dir):
    os.mkdir(dir)

# Сохраняем изображения в папку в формате .png:
for i in range(len(images_jpg)):
    cv.imwrite(f'{dir}/{filename[i]}.png', s[i])
