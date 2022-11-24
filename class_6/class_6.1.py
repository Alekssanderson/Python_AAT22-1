# Решим предыдущую задачу с помощью MLPClassifier

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


data = pd.ExcelFile("data.xlsx")
data_parse = data.parse('Лист1')

learn_array = np.hstack((data_parse.values[2:, 4:7], data_parse.values[2:, 8:11])).astype(float)
# Желаемый результат:
output_values = data_parse.values[2:, 3:4].astype(int).flatten()

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
Желаемый результат = 15, прогноз = 15
Желаемый результат = 11, прогноз = 11
Желаемый результат = 20, прогноз = 20
Желаемый результат = 7, прогноз = 7
Желаемый результат = 15, прогноз = 15
Желаемый результат = 7, прогноз = 7
Желаемый результат = 13, прогноз = 13
Желаемый результат = 11, прогноз = 11
Желаемый результат = 13, прогноз = 13
Желаемый результат = 20, прогноз = 20
Желаемый результат = 5, прогноз = 5
Желаемый результат = 10, прогноз = 10
Желаемый результат = 8, прогноз = 8
Желаемый результат = 12, прогноз = 12
Желаемый результат = 10, прогноз = 10
Желаемый результат = 8, прогноз = 8
Желаемый результат = 12, прогноз = 12
Желаемый результат = 20, прогноз = 20
Желаемый результат = 15, прогноз = 15
Желаемый результат = 13, прогноз = 13
Желаемый результат = 13, прогноз = 13
Желаемый результат = 5, прогноз = 5
Желаемый результат = 8, прогноз = 8
Желаемый результат = 8, прогноз = 8
Желаемый результат = 20, прогноз = 20
Желаемый результат = 15, прогноз = 15
Желаемый результат = 15, прогноз = 15

Все совпадает, отклонений нет.
"""

# Протестируем наш нейрон, используя новые данные:
data_test = pd.ExcelFile("data_test.xlsx")
data_test_parse = data_test.parse('Лист1')

array_test = np.hstack((data_test_parse.values[2:, 4:7], data_test_parse.values[2:, 8:11])).astype(float)
# Желаемый результат:
output_values_test = data_test_parse.values[2:, 3:4].flatten()

# Нормализуем данные array_test:
for i in range(len(array_test[0])):
    b = np.polyfit(np.sort(array_test[:, i]), np.linspace(0, 1, len(array_test[:, i])), 1)
    array_test[:, i] = array_test[:, i] * b[0] + b[1]

predict_data_test = clf.predict(array_test)

for i in range(len(output_values_test)):
    print(f'Желаемый результат = {output_values_test[i]}, прогноз = {predict_data_test[i]}')

"""
Желаемый результат = 11, прогноз = 13
Желаемый результат = 7, прогноз = 7
Желаемый результат = 12, прогноз = 20
Желаемый результат = 10, прогноз = 10
Желаемый результат = 15, прогноз = 20

Результат почти схож с прошлой задачей, где эти же самые данные тестировали в однослойном перцептроне:
    11 -> 15.595
    7 -> 8.731
    12 -> 18.678
    10 -> 14.562
    15 -> 18.972
"""
