# -*- coding: utf-8 -*-
"""
ml контрольная
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# читаем данные из файла
data = pd.read_csv('C:\prog\data.csv')

# выводим данные
data.head()
# смотрим типы данных
data.info()

# Меняем строку notAvalibal на nan 
data = data.replace({'Not Available': np.nan})


# Гистограмма распределения объема реализации бензина
plt.style.use('fivethirtyeight')
plt.hist(data['бензин'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Объем реализации (т)'); plt.ylabel('Количество заправок'); 
plt.title('Объем реализации бензина');

# Создание списка АЗС с более чем 10 записями для вывода в график
types = data.dropna(subset=['бензин'])
types = types['Заправки'].value_counts()
types = list(types[types.values > 10].index)

# График распределения реализации по АЗС
plt.figure(figsize = (10, 9))

# График каждой АЗС
for b_type in types:
    # Выбираем все заправки
    subset = data[data['Заправки'] == b_type]
    
    # График плотности реализации бензина по АЗС
    sns.kdeplot(subset['бензин'].dropna(),
               label = b_type, shade = False, alpha = 0.8);
    
# Наименования к графику по осям
plt.xlabel('Объем реализации бензина', size = 10); plt.ylabel('Плотность', size = 10); 
plt.title('График плотности реализации бензина по АЗС', size = 14);
    
# Нахождение всех корелирующих фич 
correlations_data = data.corr()['   автобензин А98'].sort_values()

