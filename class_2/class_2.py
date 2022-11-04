import numpy as np

# 1. Создаем электронную таблицу в Excel, сохраняем в формате .csv
# 2. Открываем электронную таблицу. Загружаем данные из электронной таблицы data.csv в массив
file = open('data.csv', 'rb')
data = np.genfromtxt(file, delimiter=';', dtype=float)
file.close()
print(data)  # таблица Пифагора от 1 до 5
# [[nan  1.  2.  3.  4.  5.]
#  [ 1.  1.  2.  3.  4.  5.]
#  [ 2.  2.  4.  6.  8. 10.]
#  [ 3.  3.  6.  9. 12. 15.]
#  [ 4.  4.  8. 12. 16. 20.]
#  [ 5.  5. 10. 15. 20. 25.]]

# Проверяем тип данных data
print(type(data))  # <class 'numpy.ndarray'>

# 3. Создаем единичную матрицу размерности, идентичной полученной из электронной таблицы.
# Определяем размерность data функцией shape
row_col = data.shape  # (6, 6)
ones_array = np.ones(row_col, dtype=int)
print(ones_array)
# [[1 1 1 1 1 1]
#  [1 1 1 1 1 1]
#  [1 1 1 1 1 1]
#  [1 1 1 1 1 1]
#  [1 1 1 1 1 1]
#  [1 1 1 1 1 1]]

# 4. Перемножаем одно на другое и экспортируем результат в электронную таблицу
data_new = data * ones_array
print(data_new)
# [[nan  1.  2.  3.  4.  5.]
#  [ 1.  1.  2.  3.  4.  5.]
#  [ 2.  2.  4.  6.  8. 10.]
#  [ 3.  3.  6.  9. 12. 15.]
#  [ 4.  4.  8. 12. 16. 20.]
#  [ 5.  5. 10. 15. 20. 25.]]
save_array = np.savetxt('save_array.csv', data_new, fmt='%.2f', delimiter=';')
