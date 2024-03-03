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

# In[7]:


train = pd.read_csv("sign_mnist_train.csv").values
letraM.values[0][1:].reshape(4, 4)


# # Visualizamos una imagen

# In[3]:

im =letraM.values[0][1:].reshape(28,28)
im2 =letraM.values[1][1:].reshape(28,28)
# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 2)

# Mostrar la primera imagen en el primer subgráfico
axes[0].matshow(im[1:], cmap="gray")
axes[0].set_title('Imagen 1')

# Mostrar la segunda imagen en el segundo subgráfico
axes[1].matshow(im2[1:], cmap="gray")
axes[1].set_title('Imagen 2')

# Ajustar el diseño para evitar superposiciones
plt.tight_layout()

# Mostrar la figura
plt.show()

#%%
import matplotlib.pyplot as plt

# Suponiendo que ya tienes las imágenes im e im2 definidas

# Obtener los valores de los píxeles de ambas imágenes
pixels_im = im.flatten()
pixels_im2 = im2.flatten()

# Crear un gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(pixels_im, pixels_im2, s=5, c='b', alpha=0.5)

# Configurar etiquetas y título
plt.xlabel('Intensidad de pixels en imagen 1')
plt.ylabel('Intensidad de pixels en imagen 2')
plt.title('Relación entre las intensidades de pixels')

# Mostrar el gráfico
plt.grid(True)
plt.show()

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
