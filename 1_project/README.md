**Goal: Predict the species of an iris flower using petal and sepal measurements.**

**Dataset: UCI ML Repository (Iris dataset)**

**Skills: Classification, scikit-learn basics**

```python
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

```

<img width="640" height="480" alt="iris_scatter" src="https://github.com/user-attachments/assets/b5993580-8e69-4e85-91c4-cc979f71c9bd" />

<img width="1280" height="677" alt="image" src="https://github.com/user-attachments/assets/77601e3d-2a60-4c63-b203-e87af9023867" />


**pandas**: Для работы с табличными данными (DataFrame).

**numpy**: Для числовых вычислений.

**matplotlib.pyplot**: Для визуализации данных.

**load_iris**: Функция из sklearn для загрузки датасета Iris.

**train_test_split**: Разделение данных на обучающую и тестовую выборки.

**StandardScaler**: Нормализация данных (масштабирование признаков).

**KNeighborsClassifier**: Алгоритм k-ближайших соседей (KNN) для классификации.

**accuracy_score, classification_report, confusion_matrix**: Метрики для оценки модели.

**matplotlib.use('Agg')**: Установка бэкэнда Matplotlib для сохранения графиков без отображения на экране.




Мы учили компьютер распознавать три вида цветков ириса: Setosa, Versicolor и Virginica. Для этого мы дали ему данные о размерах лепестков и чашелистников этих цветов, а потом проверили, сможет ли он правильно угадывать вид цветка, когда ему дают новые данные.

То есть в итоге мы загрузили датасет Iris, подготавливили данные (разделение на признаки и целевую переменную, нормализация), создали и обучает модель KNN, проверили качество модели на тестовых данных, визуализировали данные и сохранили график.

Изначально у нас был набор данных с 150 цветками ириса. Для каждого цветка было четыре числа:

1) Длина чашелистника (sepal length).
2) Ширина чашелистника (sepal width).
3) Длина лепестка (petal length).
4) Ширина лепестка (petal width).
5) И ещё одно число — это был вид цветка (Setosa = 0, Versicolor = 1, Virginica = 2).


Далее мы разделили данные на две части:

Обучающая выборка (80% данных): Это те данные, которые мы дали компьютеру для обучения.
Тестовая выборка (20% данных): Это те данные, которые мы использовали для проверки, насколько хорошо компьютер научился.



Потом мы использовали метод под названием KNN (k-ближайших соседей). Этот метод работает так:

Когда компьютеру нужно определить вид нового цветка, он смотрит на три ближайших цветка из обучающей выборки (потому что n_neighbors=3).
Он смотрит, к какому виду принадлежат эти три ближайших цветка, и выбирает самый популярный вид среди них.
Например, если два ближайших цветка — это Setosa, а один — Versicolor, то новый цветок будет классифицирован как Setosa.


После обучения мы дали компьютеру тестовые данные и попросили его предсказать виды цветков. Затем мы сравнили его ответы с правильными ответами и посмотрели, насколько он был прав. Результат показал, что компьютер угадал все цветки правильно. 

Был нарисован график, чтобы лучше понять, как выглядят данные.

Ось X — это длина лепестка.

Ось Y — это ширина лепестка.

Разные цвета точек обозначают разные виды цветков разных видов, которые имеют разные размеры лепестков.

