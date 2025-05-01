import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, InputLayer

# Функция для one-hot кодирования последовательности ДНК
def one_hot_encode_seq(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq.upper()):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    return one_hot

# Функция генерации синтетических последовательностей:
# Для положительных примеров вставляется заданный мотив.
def generate_synthetic_data(num_samples, seq_length, motif, pos_fraction=0.5):
    sequences = []
    labels = []
    for _ in range(num_samples):
        # Генерируем случайную последовательность длины seq_length
        random_seq = ''.join(random.choice('ACGT') for _ in range(seq_length))
        if random.random() < pos_fraction:  # Положительный пример
            start = random.randint(0, seq_length - len(motif))
            seq = random_seq[:start] + motif + random_seq[start+len(motif):]
            label = 1
        else:  # Отрицательный пример
            seq = random_seq
            label = 0
        sequences.append(seq)
        labels.append(label)
    return sequences, np.array(labels)

# Определение свёрточной модели для работы с one-hot представлением ДНК.
# Функция принимает длину последовательности (input_length) и число классов (num_classes).
def create_cnn_model(input_length, num_classes):
    model = Sequential([
        # Первый свёрточный слой:
        # Извлекает локальные признаки из входного one-hot представления ДНК.
        # Использует 16 фильтров, размер ядра 4 и активацию ReLU.
        Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(input_length, 4)),

        # MaxPooling слой:
        # Уменьшает размерность, оставляя наиболее важные признаки в каждом окне (pool_size=2).
        MaxPooling1D(pool_size=2),

        # Второй свёрточный слой:
        # Дополнительно извлекает особенности, используя 16 фильтров и размер ядра 4.
        Conv1D(filters=16, kernel_size=4, activation='relu'),

        # Второй MaxPooling слой:
        # Снова уменьшает размерность для сокращения вычислений и улучшения обобщения.
        MaxPooling1D(pool_size=2),

        # Слой Flatten:
        # Преобразует полученные двумерные признаки в одномерный вектор для последующих слоёв.
        Flatten(),

        # Полносвязный слой Dense:
        # Содержит 100 нейронов с активацией ReLU для изучения высокоуровневых представлений.
        Dense(100, activation='relu'),

        # Слой Dropout:
        # Применяет регуляризацию с вероятностью отключения нейронов 0.5 для снижения переобучения.
        Dropout(0.5),

        # Выходной слой Dense:
        # Количество нейронов равно числу классов, активация softmax выдаёт распределение вероятностей по классам.
        Dense(num_classes, activation='softmax')
    ])

    # Компиляция модели: Adam-оптимизатор, функция потерь для классификации с метками в формате int
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Параметры для синтетических данных
seq_length = 1000
motif = 'ACGTACGT'  # Достоверный мотив для положительных примеров
num_samples = 100
num_classes = 2

# Генерируем синтетический набор данных
dna_sequences, labels = generate_synthetic_data(num_samples, seq_length, motif, pos_fraction=0.5)
print("Пример последовательности:", dna_sequences[0])
print("Метки:", labels[:10])

# Преобразуем последовательности в one-hot представление
X = np.array([one_hot_encode_seq(seq) for seq in dna_sequences])
print("Форма X:", X.shape)  # Ожидается (num_samples, seq_length, 4)

# Разделяем данные на обучающую и тестовую выборки (например, 80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# Создаём модель
model = create_cnn_model(input_length=seq_length, num_classes=num_classes)
model.summary()

# Обучаем модель на обучающем наборе
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Оцениваем модель на тестовой выборке
score = model.evaluate(X_test, y_test)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])