#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inline_sql import sql, sql_val
from sklearn.decomposition import PCA
import seaborn as sns
#%%
#Cargamos archivo csv

carpeta = './datasets/'
sign = pd.read_csv(carpeta+'sign_mnist_train.csv')

#%%
#Comenzamos explorando, conociendo cuantas muestras poseemos de cada letra

#Los labels van de 0 a 24, no incluye el 9 (la j) ni el 25(la z)
#armamos un df con la informacion de los casos de test que tenemos

abc = pd.DataFrame({'letras': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 
                               'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'],
                    'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 
                              17, 18, 19, 20, 21, 22, 23, 24]})


#ahora agrupamos para obtener la cantidad de muestras que tenemos por cada letra

cantMuestras= sql^"""SELECT s.label, COUNT(*) as cantidad, abc.letras
            FROM sing as s
            INNER JOIN abc
            ON abc.label = d.label
            GROUP BY s.label, abc.letras
            ORDER BY s.label;
            """
#graficamos lo obtenido 
fig, ax = plt.subplots()

ax.bar(x=cantMuestras['letras'], height=cantMuestras['cantidad'],color = 'purple')
ax.set_title ('Cantidad de muestras por letra')
ax.set_xlabel ('Letras')
ax.set_ylabel ('Cantidad de muestras')

#%%
#Separamos tres dataframes de letras para comparar
letraM = sign[sign['label']==12]

letraE = sign[sign['label']==4]

letraL = sign[sign['label']==11]


# # Convertimos a array el dataframe y nos quedamos con una foto en formato (28x28)


#%% Exploración visual comparativa de dos imagenes de una misma letra

im =letraM.values[0][1:].reshape(28,28)
im2 =letraM.values[1][1:].reshape(28,28)
im3 =letraM.values[25][1:].reshape(28,28) #similar a im pero más rotada hacia atrás y con más iluminación
im4 =letraM.values[19][1:].reshape(28,28) #similar a im2 pero más oscura y menos nítida

# Crear una figura con dos subgráficos
fig, ax = plt.subplots(1, 4)

# Mostrar la primera imagen en el primer subgráfico
ax[0].matshow(im[1:], cmap="gray")
ax[0].set_title('Imagen 1')

# Mostrar la segunda imagen en el segundo subgráfico
ax[1].matshow(im2[1:], cmap="gray")
ax[1].set_title('Imagen 2')

# Mostrar la tercera imagen en el tercer subgráfico
ax[2].matshow(im3[1:], cmap="gray")
ax[2].set_title('Imagen 3')

# Mostrar la cuarta imagen en el cuarto subgráfico
ax[3].matshow(im4[1:], cmap="gray")
ax[3].set_title('Imagen 4')

plt.suptitle("Comparación de imagenes de la letra M", y=0.3)
# Ajustar el diseño para evitar superposiciones
plt.tight_layout()
plt.show()

"""
En esta comparación de los datos correspondientes a las filas 0, 1, 3 y 19 
del dataframe de la letra M, podemos observar que las imagenes de una misma letra varían 
en ángulo (imagen 1 y 3) y luminosidad y nitidez (imagen 2 y 4).
"""
#%% Exploración visual para comparar en qué valores de intensidad se concentran los
# pixeles de las imagenes 2 y 4, que son similares pero varían en luminosidad.  

# Obtener los valores de los píxeles de ambas imágenes
pixels_im2 = im2.reshape(1,784)
pixels_im4 = im4.reshape(1,784)

# Crear un gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(pixels_im2, pixels_im4, s=3, c='g', label = 'imagen 2')
plt.scatter(pixels_im4, pixels_im2, s=3, c='b', label = 'imagen 4')

# Configurar etiquetas y título
plt.xlabel('Intensidad de pixels en imagen 2')
plt.ylabel('Intensidad de pixels en imagen 4')
plt.title('Relación entre las intensidades de pixels')

# Mostrar el gráfico
plt.grid(True)
plt.show()

"""
En esta comparación se puede observar cómo la intensidad de los pixeles de ambas imagenes 
se distribuye de manera similar (es decir cómo los puntos se encuentran espaciados)
pero con cierta traslación en el plano.
"""
#%% Exploración visual para comparar en qué valores de intensidad se concentran los
# pixeles de las imagenes 1 y 3, que son similares pero varían en rotación y luminosidad.  

# Obtener los valores de los píxeles de ambas imágenes
pixels_im = im.reshape(1,784)
pixels_im3 = im3.reshape(1,784)

# Crear un gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(pixels_im, pixels_im3, s=3, c='g', label = 'imagen 1')
plt.scatter(pixels_im3, pixels_im, s=3, c='b', label = 'imagen 3')

# Configurar etiquetas y título
plt.xlabel('Intensidad de pixels en imagen 1')
plt.ylabel('Intensidad de pixels en imagen 3')
plt.title('Relación entre las intensidades de pixels')

# Mostrar el gráfico
plt.grid(True)
plt.show()

"""
En esta comparación se puede observar cómo la intensidad de los pixeles de ambas imagenes 
parece reflejarse, lo que implica que hay cierta relación inversa entre las intensidades
(reflexión respecto a la recta y=x). 
"""

#%%

im =letraM.values[0][1:].reshape(28,28)
im2 =letraM.values[1][1:].reshape(28,28)  

# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 2)

# Mostrar la primera imagen en el primer subgráfico
axes[0].matshow(im[1:2], cmap="gray")
axes[0].set_title('Imagen 1')

# Mostrar la segunda imagen en el segundo subgráfico
axes[1].matshow(im2[1:2], cmap="gray")
axes[1].set_title('Imagen 2')

# Ajustar el diseño para evitar superposiciones
plt.tight_layout()

# Mostrar la figura
plt.show()
#%%
#Ahora exploramos visualmente como varía el comportamiento de los datos, cuando las letras son muy parecidas entre sí y cuando son distintas
#Comparamos dos letras similares visualmente
data_E_M = sql^"""
                select *
                from sing
                where label == 4 or label == 12
"""

X = data_E_M.drop('label', axis=1).values
y = data_E_M['label'].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[y == 4, 0], X_pca[y == 4, 1], label='Label 4', s=7)
plt.scatter(X_pca[y == 12, 0], X_pca[y == 12, 1], label='Label 12', s=7)

plt.title('PCA : comparar E y M')
plt.xlabel('Primera Componente')
plt.ylabel('Segunda Componente')
plt.show()

#Ahora comparamos dos letras totalmente distintas visualmente
data_E_L = sql^"""
                select *
                from sing
                where label == 4 or label == 11
"""

X = data_E_L.drop('label', axis=1).values
y = data_E_L['label'].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[y == 4, 0], X_pca[y == 4, 1], label='Label 4', s=7)
plt.scatter(X_pca[y == 11, 0], X_pca[y == 11, 1], label='Label 11', s=7)  # Corrección aquí


plt.title('PCA : comparar E y L ')
plt.xlabel('Primera Componente')
plt.ylabel('Segunda Componente')
plt.show()
