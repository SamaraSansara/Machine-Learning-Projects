import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')


iris = load_iris() #Загрузка датасета


# Преобразование данных в DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
# Мы преобразуем массив данных из iris.data в таблицу (DataFrame) с помощью pandas.
# Добавляем новый столбец species, который содержит целевые метки (классы цветков).

print(df.head()) # Эта команда выводит первые 5 строк таблицы


# Разделение данных на признаки и целевую переменную
X = df.drop('species', axis=1) # Матрица признаков (все столбцы, кроме species).
y = df['species'] # Матрица признаков (все столбцы, кроме species).


# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Обучающая выборка (X_train, y_train) — 80% данных.
# Тестовая выборка (X_test, y_test) — 20% данных.


scaler = StandardScaler() # нормализует данные, приводя их к стандартному виду (среднее = 0, стандартное отклонение = 1).
X_train = scaler.fit_transform(X_train) # Применяется к обучающей выборке для обучения масштабировщика.
X_test = scaler.transform(X_test) # Применяется к тестовой выборке для использования уже обученного масштабировщика.



# Мы создаём модель KNN с параметром n_neighbors=3 (используется 3 ближайших соседа для классификации).
# Обучаем модель на обучающих данных (X_train, y_train).
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)  # Модель предсказывает классы для тестовых данных (X_test). Результат хранится в y_pred.

accuracy = accuracy_score(y_test, y_pred) #  Сравнивает истинные значения (y_test) с предсказанными (y_pred) и вычисляет точность.
print(f"Точность модели: {accuracy * 100:.2f}%")
print("Отчет по классификации:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(conf_matrix)

plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['species'], cmap='viridis')
plt.xlabel('Длина лепестка (см)')
plt.ylabel('Ширина лепестка (см)')
plt.title('Распределение видов ирисов')
plt.savefig("iris_scatter.png")
