# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as skl


    
    # Считываем данные 
train_features = pd.read_csv('training_features.csv')
test_features = pd.read_csv('test_features.csv')
train_labels = pd.read_csv('training_label.csv')
test_labels = pd.read_csv('test_label.csv')


# смотрим типы данных
train_features.info()

#заполняем медианными значениями пробелы(пустоты)
imputer = skl.Imputer(strategy='median')
imputer.fit(train_features)

X = imputer.transform(train_features)
X_test = imputer.transform(test_features)

y = np.array(train_labels).reshape((-1, ))
y_test = np.array(test_labels).reshape((-1, ))

#градиентный бустинг
from sklearn.ensemble import GradientBoostingRegressor

# Создаем модель
gradient_boosted = GradientBoostingRegressor()

# учим модель на обучающей выборке
gradient_boosted.fit(X, y)

# делаем прогнозы по тестовой выборке
predictions = gradient_boosted.predict(X_test)

# оцениваем модель
mae = np.mean(abs(predictions - y_test))

print('Gradient Boosted Performance on the test set: MAE = %0.4f' % mae)

plt.figure(figsize = (6, 5))

# Density plot of the final predictions and the test values
sns.kdeplot(predictions, label = 'Прогноз')
sns.kdeplot(y_test, label = 'Значение')

# Label the plot
plt.xlabel('Реализация А98'); plt.ylabel('Плотность');
plt.title('Сравнение реальных данных и предсказаний');
    