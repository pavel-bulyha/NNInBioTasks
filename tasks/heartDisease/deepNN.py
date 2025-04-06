import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загружаем данные
df = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

# Разделение на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализуем данные
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Построение модели с подобранными гиперпараметрами:
# units_1: 128, dropout_1: 0.3, units_2: 64, dropout_2: 0.3, learning_rate: 0.00059063
model = Sequential([

    # Слой 1
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.30000000000000004),

    # Слой 2
    Dense(64, activation='relu'),
    Dropout(0.30000000000000004),

    # Выходной слой
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0005906296261520694),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train,
                    epochs=8,
                    batch_size=4,
                    validation_data=(X_test, y_test))
