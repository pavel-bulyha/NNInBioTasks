# ============================
# Альтернативная версия построения графиков (мой стиль)
# ============================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris

# Настройка стиля seaborn для более современного вида графиков
sns.set(style="whitegrid", context="notebook")

# Загрузка и подготовка данных
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target_df = pd.DataFrame(data=iris.target, columns=["species"])

# Функция для преобразования числовых меток в текстовые классы
def converter(specie):
    return ["setosa", "versicolor", "virginica"][specie]

target_df["species"] = target_df["species"].apply(converter)
iris_df_graf = pd.concat([iris_df, target_df], axis=1)

# Назначение цветов для разных видов
colors = {"setosa": "red", "versicolor": "blue", "virginica": "green"}

# 1. АЛЬТЕРНАТИВА: Гистограммы распределения признаков с использованием seaborn
plt.figure(figsize=(12, 8))
for i, col in enumerate(iris_df.columns, start=1):
    plt.subplot(2, 2, i)
    sns.histplot(data=iris_df_graf, x=col, hue="species",
                 palette=colors, kde=True, element="step")
    plt.title(f"Распределение {col}")
plt.tight_layout()
plt.show()

# Альтернативный вариант гистограмм (закомментирован):
# iris_df_graf.hist(column=iris_df.columns, figsize=(12,8), grid=False)
# plt.suptitle("Гистограммы распределения признаков")
# plt.show()

# 2. АЛЬТЕРНАТИВА: Попарный анализ зависимостей с помощью seaborn pairplot
sns.pairplot(iris_df_graf, hue="species", palette=colors)
plt.suptitle("Попарные диаграммы рассеяния для набора ирисов", y=1.02)
plt.show()

# Альтернативный вариант scatter plot по подграфикам (закомментирован):
# plt.figure(figsize=(10,10))
# plot_idx = 1
# for i in range(4):
#     for j in range(i+1, 4):
#         plt.subplot(3, 2, plot_idx)
#         sns.scatterplot(data=iris_df_graf, x=iris_df.columns[i], y=iris_df.columns[j],
#                         hue="species", palette=colors, alpha=0.8)
#         plt.title(f"{iris_df.columns[i]} vs {iris_df.columns[j]}")
#         plot_idx += 1
# plt.tight_layout()
# plt.show()

# 3. АЛЬТЕРНАТИВА: Boxplot для каждого признака с использованием seaborn
plt.figure(figsize=(12, 8))
for i, col in enumerate(iris_df.columns, start=1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="species", y=col, data=iris_df_graf, palette=colors)
    plt.title(f"Boxplot по {col}")
plt.tight_layout()
plt.show()

# Альтернативный вариант boxplot (с использованием pandas, закомментирован):
# iris_df_graf.boxplot(column=iris_df.columns.tolist(), by="species", figsize=(12,8))
# plt.title("Boxplot по видам")
# plt.suptitle("")
# plt.show()

# 4. АЛЬТЕРНАТИВА: Корреляционная матрица с использованием seaborn heatmap
correlation_matrix = iris_df_graf.drop(columns=["species"]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Корреляционная матрица")
plt.show()

# Альтернативный вариант для корреляционной матрицы (закомментирован):
# plt.figure(figsize=(8,6))
# plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
# plt.colorbar()
# plt.xticks(range(len(iris_df.columns)), iris_df.columns, rotation=90)
# plt.yticks(range(len(iris_df.columns)), iris_df.columns)
# plt.title("Корреляционная матрица")
# plt.show()
