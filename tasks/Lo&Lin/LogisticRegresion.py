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
# Разделение данных на обучающую и тестовую выборки
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
print('данные тренировочные (размер):', X_train.shape, 'данные тестовые (размер):',X_test.shape)
# 33% данных выделяем на тестирование
model_log_reg = LogisticRegression(max_iter=200) # Создание экземпляра модели логистической регрессии
print('параметры модели:', model_log_reg.get_params()) # метод get_params - вызывает информацию об параметрах модели
model_log_reg.fit(X_train, y_train) #метод fit - обучает модель на обучающей (трейн) выборке
# Предсказания
predicted_train = model_log_reg.predict(X_train)
predicted_test = model_log_reg.predict(X_test) # метод predict - показывает ответы модели на тестовой выборке
print('предсказание модели на тренировочном наборе:\n', predicted_train, '\nпредсказание модели на тестовом наборе:\n', predicted_test)
score = model_log_reg.score(X_test, y_test)
print(f"Score method - Accuracy: {score}")
accuracy = accuracy_score(y_test, predicted_test)
print(f"Accuracy: {accuracy:.2f}")
# Матрица ошибок
conf_matrix = confusion_matrix(y_test, predicted_test)
print("Матрица ошибок:")
print(conf_matrix)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Матрица ошибок для классификации видов ириса')
plt.show()