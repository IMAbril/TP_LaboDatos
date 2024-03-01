# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

carpeta ='/carpetita/'

# In[1]:

#Importamos el archivo csv como un dataFrame
train2 = pd.read_csv(carpeta +"sign_mnist_train.csv")

#Imprimimos las primeras filas
train2.head()

# In[2]:

#Ahora convertimos el dataFrame en un array de NumPy, con el atributo .values() y nos quedamos con una foto en formato (28x28)

train = pd.read_csv(carpeta + "sign_mnist_train.csv").values
train[0][1:].reshape(28, 28)

# In[3]:
#Ahora visualizamos una imagen, omitiendo el nombre de la columna con [:1]
plt.matshow(train[0][1:].reshape(28, 28), cmap = "gray")
