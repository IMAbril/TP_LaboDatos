#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:52:07 2024

@author: abril
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

carpeta = './datasets/'
sign = pd.read_csv(carpeta+'sign_mnist_train.csv')


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


#título
plt.suptitle("Comparación de imagenes de la letra M", y=0.3)

# Ajustar el diseño para evitar superposiciones
plt.tight_layout()

# Mostrar la figura
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
