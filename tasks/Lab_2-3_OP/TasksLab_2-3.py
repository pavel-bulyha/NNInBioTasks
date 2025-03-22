import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import missingno as msno

# Задание 1.1
randArr = np.random.rand(10,10)
maxValue = np.max(randArr)
print(randArr)
print(maxValue)
# Задание 1.2
zerosArr = np.zeros((10,10))
zerosArr[1:-1,1:-1] = 1
print(zerosArr)
# Задание 2.1
pd.set_option('display.max_columns', None)
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target_df = pd.DataFrame(data=iris.target, columns=['species'])
iris_df = pd.concat([iris_df, target_df], axis=1)
IrCorr = iris_df.corr()
print(iris_df)
print(IrCorr)
print("max corr: ", IrCorr.values[IrCorr.values != 1].max())
print("min corr: ", IrCorr.values.min())

"""
Petal length (cm) и petal width (cm): Корреляция равна 0.962865 — это самый высокий показатель, 
за исключением диагональных единиц. 
Такой высокий коэффициент указывает на почти линейную зависимость: 
когда длина лепестка увеличивается, ширина лепестка также практически пропорционально увеличивается.

Связь с species: Также стоит отметить, что признаки, связанные с размерами лепестков, очень сильно коррелируют с меткой вида (species):
petal length — 0.949035
petal width — 0.956547 Это говорит о том, что размеры лепестков являются хорошими признаками для определения вида ириса.

Sepal length (cm) и sepal width (cm): Корреляция между ними составляет -0.117570, что по модулю очень мало. 
Это означает, что между длиной и шириной чашелистика практически отсутствует линейная связь: 
изменение одного признака почти никак не связано с изменением другого.

Как применять на практике, нуууу, можно определять вид по лепесткам:)
"""

# Задание 2.2
print("Не обновленный DataFrame ")
print(iris_df.describe())
np.random.seed(42)
missing_indices = np.random.choice(iris_df.index, size=5, replace=False)
iris_df.loc[missing_indices, np.random.choice(iris_df.columns)] = np.nan


duplicates = iris_df.sample(3, random_state=42)
iris_df = pd.concat([iris_df, duplicates], ignore_index=True)


outlier_indices = np.random.choice(iris_df.index, size=3, replace=False)
iris_df.loc[outlier_indices, np.random.choice(iris_df.columns)] *= 3

# Вывод информации о данных
print("Обновленный DataFrame ")
print(iris_df.describe())
msno.matrix(iris_df)
plt.show()

"""
В обновленном наборе данных по сравнению с исходным df_iris появились три основные проблемы:

Пропущенные значения (NaN), вызванные заменой случайных ячеек на np.nan.

Дублирование строк, что искажает статистические свойства датасета.

Выбросы, вызванные умножением значений на 3 в ряде строк, что приводит к экстремальным значениям.

Для исправления этих проблем нужно модифицировать датасет:

Заполнить или удалить пропущенные значения (например, с помощью fillna() или dropna()).

Удалить дубликаты с помощью метода drop_duplicates().

Обнаружить и обработать выбросы с использованием методов статистической фильтрации (например, по методу IQR или с помощью Z-score).
"""
# применение методов на практике
    # Импутация (заполнение) NaN:
for col in iris_df.select_dtypes(include=[np.number]).columns:
    iris_df[col].fillna(iris_df[col].median())

    # Удаление дубликатов
iris_df = iris_df.drop_duplicates()

    # Замена выбросов медианой:
for col in iris_df.select_dtypes(include=[np.number]).columns:
    Q1 = iris_df[col].quantile(0.25)
    Q3 = iris_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Заменяем выбросы на медиану столбца
    iris_df.loc[(iris_df[col] < lower_bound) | (iris_df[col] > upper_bound), col] = iris_df[col].median()

print("Обновленный и исправленный DataFrame ")
print(iris_df.describe())
msno.matrix(iris_df)
plt.show()