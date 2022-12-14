"""
При помощи MLPClassifier решить задачу
определения цены позиции из прайс-листа.

Посмотреть, как меняется результат
в зависимости от hidden_layer_sizes
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


data = pd.ExcelFile("data.xlsx")
data_parse = data.parse('Лист1')

learn_array = np.hstack((data_parse.values[2:, 3:7], data_parse.values[2:, 8:10])).astype(float)
# Желаемый результат:
output_values = data_parse.values[2:, 10:11].astype(float).flatten()  # [ 87.  71. 137.  ...  95.  76. 193.]

# Нормализуем данные learn_array:
for i in range(len(learn_array[0])):
    a = np.polyfit(np.sort(learn_array[:, i]), np.linspace(0, 1, len(learn_array[:, i])), 1)
    learn_array[:, i] = learn_array[:, i] * a[0] + a[1]

clf = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(200,),
    random_state=1,
    max_iter=300,
    warm_start=True
)

clf.fit(learn_array, output_values)
predict_data = clf.predict(learn_array)

for i in range(len(output_values)):
    print(f'Желаемый результат = {output_values[i]}, прогноз = {predict_data[i]}')

"""
Желаемый результат = 87.0, прогноз = 87.0
Желаемый результат = 71.0, прогноз = 71.0
Желаемый результат = 137.0, прогноз = 137.0
Желаемый результат = 75.0, прогноз = 75.0
Желаемый результат = 115.0, прогноз = 115.0
Желаемый результат = 305.0, прогноз = 305.0
Желаемый результат = 99.0, прогноз = 99.0
Желаемый результат = 52.0, прогноз = 52.0
Желаемый результат = 270.0, прогноз = 270.0
Желаемый результат = 174.0, прогноз = 174.0
Желаемый результат = 62.0, прогноз = 62.0
Желаемый результат = 75.0, прогноз = 75.0
Желаемый результат = 38.0, прогноз = 38.0
Желаемый результат = 293.0, прогноз = 293.0
Желаемый результат = 345.0, прогноз = 345.0
Желаемый результат = 128.0, прогноз = 128.0
Желаемый результат = 225.0, прогноз = 225.0
Желаемый результат = 117.0, прогноз = 117.0
Желаемый результат = 156.0, прогноз = 115.0  --- НЕ СОВПАДАЕТ
Желаемый результат = 128.0, прогноз = 128.0
Желаемый результат = 172.0, прогноз = 172.0
Желаемый результат = 71.0, прогноз = 71.0
Желаемый результат = 67.0, прогноз = 67.0
Желаемый результат = 107.0, прогноз = 107.0
Желаемый результат = 95.0, прогноз = 95.0
Желаемый результат = 76.0, прогноз = 76.0
Желаемый результат = 193.0, прогноз = 193.0

Почти все совпадает.
"""

# Протестируем наш нейрон, используя новые данные:
data_test = pd.ExcelFile("data_test.xlsx")
data_test_parse = data_test.parse('Лист1')

array_test = np.hstack((data_test_parse.values[2:, 3:7], data_test_parse.values[2:, 8:10])).astype(float)
# Желаемый результат:
output_values_test = data_test_parse.values[2:, 10:11].astype(float).flatten()

# Нормализуем данные array_test:
for i in range(len(array_test[0])):
    b = np.polyfit(np.sort(array_test[:, i]), np.linspace(0, 1, len(array_test[:, i])), 1)
    array_test[:, i] = array_test[:, i] * b[0] + b[1]

predict_data_test = clf.predict(array_test)
print("---")
for i in range(len(output_values_test)):
    print(f'Желаемый результат = {output_values_test[i]}, прогноз = {predict_data_test[i]}')

"""
Желаемый результат = 107.0, прогноз = 225.0
Желаемый результат = 130.0, прогноз = 75.0
Желаемый результат = 75.0, прогноз = 76.0
Желаемый результат = 119.0, прогноз = 345.0
Желаемый результат = 91.0, прогноз = 137.0

Как и ранее писал в прошлой задаче:
    Причины таких больших расхождений: 
    - недостаточное количество данных; 
    - недостоверные данные в связи с тем, что сейчас в самом разгаре "черная пятница", 
      цены на некоторые позиции могут значительно снижать;
    - отсутствие какой-либо корреляции (зависимости) между характеристиками продукта;
    - недостаточное количество характеристик продукта (в нашем случае использованы только 6 характеристик 
      продукта, по факту, их гораздо больше (вес, производитель, напряжение, вид, тип колбы, форма, 
      эквивалент лампы накаливания, угол рассеивания и пр.)
"""



#______________________________________________________________________________________
"""
Меняем параметр hidden_layer_sizes в MLPClassifier и смотрим как меняется результат:
При hidden_layer_sizes = (1,) все прогнозы имеют один выход. Тестовые данные - аналогично:
    ...
    Желаемый результат = 71.0, прогноз = 128.0
    Желаемый результат = 67.0, прогноз = 128.0
    Желаемый результат = 107.0, прогноз = 128.0
    ...
 
При hidden_layer_sizes = (2,) количество выходов значений уже порядка 8 шт.
Также выводит ошибку о недостаточном количестве итераций max_iter:
    
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT. ...
    ...
    Желаемый результат = 75.0, прогноз = 75.0
    Желаемый результат = 115.0, прогноз = 137.0
    Желаемый результат = 305.0, прогноз = 71.0
    ...

При hidden_layer_sizes = (5,) почти нет отклонений, только 1 значение отклоняется.
Также выводит ошибку о недостаточном количестве итераций max_iter.
Тестовые данные и прогноз совершенно не совпадают, даже близко.

При hidden_layer_sizes = (10,) и более - почти нет отклонений, только 1 значение отклоняется.
Ошибку о недостаточном количестве итераций max_iter больше не выводит.
Тестовые данные при hidden_layer_sizes = (10,) совершенно не совпадают, даже близко.
При увеличении значения hidden_layer_sizes отклонения между тестовым результатом и прогнозом становятся меньше.
"""
