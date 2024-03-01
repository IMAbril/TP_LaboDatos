# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:02:01 2024

@author: Roju2
"""

import pandas as pd
import matplotlib.pyplot as plt
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt 

#Empezamos a explorar para comprender los datos
carpeta ='/Users/Roju2/OneDrive/Desktop/tp2/'


#Los labels van de 0 a 24, no incluye el 9 (la j) ni el 25(la z)
#armamos un df con la infromacion de los casos de test que tenemos

data =  pd.read_csv(carpeta +"sign_mnist_train.csv")



abc = pd.DataFrame({'letras': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 
                               'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'],
                    'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 
                              17, 18, 19, 20, 21, 22, 23, 24]})


#ahora agrupamos para obtener la cantidad de muestras que tenemos por cada letra

cantMuestras= sql^"""SELECT d.label, COUNT(*) as cantidad, abc.letras
            FROM data as d
            INNER JOIN abc
            ON abc.label = d.label
            GROUP BY d.label, abc.letras
            ORDER BY d.label;
            """
#graficamos lo obtenido 
fig, ax = plt.subplots()

ax.bar(x=cantMuestras['letras'], height=cantMuestras['cantidad'],color = 'purple')
ax.set_title ('Cantidad de muestras por letra')
ax.set_xlabel ('Letras')
ax.set_ylabel ('Cantidad de muestras')
