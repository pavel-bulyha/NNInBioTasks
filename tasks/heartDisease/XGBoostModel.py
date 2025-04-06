import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

# Загрузка данных
df = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")

# Разделяем признаки и целевую переменную
X = df.drop("target", axis=1)
y = df["target"]

# Разбиваем данные на обучающую и тестовую выборки.
# stratify=y гарантирует, что распределение классов останется примерно одинаковым в обеих выборках.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Инициализация модели XGBoost.
# Подобранные гиперпараметры можно корректировать под конкретную задачу.
model = xgb.XGBClassifier(
    n_estimators=500,         # число деревьев
    max_depth=8,              # максимальная глубина каждого дерева
    learning_rate=0.001,       # скорость обучения
    subsample=0.8,            # подвыборка строк для обучения каждого дерева
    colsample_bytree=0.8,     # подвыборка признаков для каждого дерева
    random_state=42,
    use_label_encoder=False,  # отключает устаревший механизм кодирования меток
    eval_metric='logloss'     # метрика для оценки качества обучения
)

# Обучение модели с ранней остановкой.
# Если на тестовой выборке в течение 10 раундов метрика не улучшается, обучение остановится.
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Расчет метрик
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
