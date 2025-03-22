import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
iris = load_iris()
# Создание фреймов данных DataFrames
iris_df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
target_df = pd.DataFrame(data= iris.target, columns= ['species'])
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'

target_df['species'] = target_df['species'].apply(converter)

# Объединить фреймы данных
iris_df = pd.concat([iris_df, target_df], axis= 1)
print('Датафрейм ирисов:\n',iris_df)
print('Датафрейм ирисов, описание:\n',iris_df.describe())
print('Датафрейм ирисов, информация:\n',iris_df.info ())
# Визуализация данных
sns.pairplot(iris_df, hue= 'species')
plt.show()
# Убираем категориальный признак species
iris_df.drop('species', axis=1, inplace=True)
print('Датафрейм ирисов после применения .drop:\n',iris_df)
# Разделение данных на обучающую и тестовую выборки
X = iris_df.drop(labels='sepal length (cm)', axis=1) # Признаки (все столбцы, кроме 'sepal length (cm)')
y = iris_df['sepal length (cm)']  # Целевая переменная (столбец 'sepal length (cm)')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
print('данные тренировочные (размер):', X_train.shape, 'данные тестовые (размер):',X_test.shape)
# 33% данных выделяем на тестирование
model = LinearRegression() # Создание экземпляра модели линейной регрессии
print('параметры модели:', model.get_params()) # метод get_params - вызывает информацию об параметрах модели
model.fit(X_train, y_train) #метод fit - обучает модель на обучающей (трейн) выборке
# Предсказания
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test) # метод predict - показывает ответы модели на тестовой выборке
print('предсказание модели на тренировочном наборе:\n', predicted_train, '\nпредсказание модели на тестовом наборе:\n', predicted_test)
# Оценка модели
print('Средняя абсолютная ошибка (MAE):', mean_absolute_error(y_test, predicted_test))
print('Средняя квадратичная ошибка (MSE):', mean_squared_error(y_test, predicted_test))
print('Среднеквадратичное отклонение (RMSE):', np.sqrt(mean_squared_error(y_test, predicted_test)))
# Визуализация предсказаний vs фактические значения
plt.scatter(y_test, predicted_test, color='blue')
plt.xlabel('Фактическая длина чашелистика (cm)')
plt.ylabel('Предсказанная длина чашелистика (cm)')
plt.title('Сравнение предсказанных и реальных значений')
plt.show()
# при уменьшении тестовой выборки ошибка становится больше, модель немного переобучается