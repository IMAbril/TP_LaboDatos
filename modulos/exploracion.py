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

carpeta = '/Users/Roju2/OneDrive/Desktop/tp2/'
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
            FROM sign as s
            INNER JOIN abc
            ON abc.label = s.label
            GROUP BY s.label, abc.letras
            ORDER BY s.label;
            """

def cantidad_De_Muestras():
        #graficamos lo obtenido 
        fig, ax = plt.subplots()
        
        ax.bar(x=cantMuestras['letras'], height=cantMuestras['cantidad'],color = 'purple')
        ax.set_title ('Cantidad de muestras por letra')
        ax.set_xlabel ('Letras')
        ax.set_ylabel ('Cantidad de muestras')
        

#%%
# =============================================================================
#  1) ANÁLISIS EXPLORATORIO: Visualización para explorar
# =============================================================================

letraE = sign[sign['label']==4]
letraM =sign[sign['label']==12]
letraL =sign[sign['label']==11]

imagen =letraE.values[0][1:].reshape(28,28)
imagen2 =letraM.values[5][1:].reshape(28,28)
imagen3 =letraL.values[9][1:].reshape(28,28)

def imagenes_Letras_PCA():
        
        # Crear una figura con tres subgráficos
        fig, ax = plt.subplots(1, 3)
        
        # Mostrar la primera imagen en el primer subgráfico
        ax[0].matshow(imagen[1:], cmap="gray")
        ax[0].set_title('Letra E')
        
        # Mostrar la segunda imagen en el segundo subgráfico
        ax[1].matshow(imagen2[1:], cmap="gray")
        ax[1].set_title('Letra M')
        
        # Mostrar la tercera imagen en el tercer subgráfico
        ax[2].matshow(imagen3[1:], cmap="gray")
        ax[2].set_title('Letra L')
        
        plt.suptitle("", y=0.2)
        # Ajustar el diseño para evitar superposiciones
        plt.tight_layout()
        plt.show()
        

#Ahora visualizamos una imagen, omitiendo el nombre de la columna con [:1]

#Exploramos visualmente como varía el comportamiento de los datos, cuando las letras son muy parecidas entre sí y cuando son distintas
def comparar_E_M():
        #Comparamos dos letras similares visualmente
        data_E_M = sql^"""
                        select *
                        from sign
                        where label == 4 or label == 12
        """
        
        X = data_E_M.drop('label', axis=1).values #Se eliminan las etiquetas de las letras de la matriz x
        y = data_E_M['label'].values #Se asignan las etiquetas al vector y 
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.scatter(X_pca[y == 4, 0], X_pca[y == 4, 1], label='Letra E', s=7, color='orange')
        plt.scatter(X_pca[y == 12, 0], X_pca[y == 12, 1], label='Letra M', s=7, color='blue')
        
        plt.title('PCA : comparar E y M')
        plt.xlabel('Primera Componente')
        plt.ylabel('Segunda Componente')
        plt.legend()
        plt.show()
        
        """
        En el gráfico cada punto representa una imagen y
        se colorean según la letra a la que corresponden (E o L).
        """
def comparar_E_L():
        #Ahora comparamos dos letras totalmente distintas visualmente
        data_E_L = sql^"""
                        select *
                        from sign
                        where label == 4 or label == 11
        """
        
        X = data_E_L.drop('label', axis=1).values
        y = data_E_L['label'].values
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.scatter(X_pca[y == 4, 0], X_pca[y == 4, 1], label='Letra E', s=5, color='orange')
        plt.scatter(X_pca[y == 11, 0], X_pca[y == 11, 1], label='Letra L', s=5, color='green')  
        
        
        plt.title('PCA : comparar E y L ')
        plt.xlabel('Primera Componente')
        plt.ylabel('Segunda Componente')
        plt.legend()
        plt.show()
        

#%%
letraC = sign[sign['label']==2]

#------Exploración visual comparativa de dos imagenes de una misma letra

# Convertimos a array el dataframe, accedemos a 3 casos y nos quedamos con 3 imagenes en formato (28x28)

im =letraC.values[0][1:].reshape(28,28)
im2 =letraC.values[5][1:].reshape(28,28)
im3 =letraC.values[9][1:].reshape(28,28)

def imagenes_Letra_C():
        
        # Crear una figura con tres subgráficos
        fig, ax = plt.subplots(1, 3)
        
        # Mostrar la primera imagen en el primer subgráfico
        ax[0].matshow(im[1:], cmap="gray")
        ax[0].set_title('Imagen 1')
        
        # Mostrar la segunda imagen en el segundo subgráfico
        ax[1].matshow(im2[1:], cmap="gray")
        ax[1].set_title('Imagen 2')
        
        # Mostrar la tercera imagen en el tercer subgráfico
        ax[2].matshow(im3[1:], cmap="gray")
        ax[2].set_title('Imagen 3')
        
        plt.suptitle("Comparación de imagenes de la letra C", y=0.2)
        # Ajustar el diseño para evitar superposiciones
        plt.tight_layout()
        plt.show()
        
def comparar_Intensidades_LetraC():
        #Comparar en qué valores de intensidad se concentran los
        # pixeles de las imagenes 1, 2 y 3 
        #--------------------------------------------------------------------
        # Obtener los valores de los píxeles de las imágenes en una fila
        pixels_im = im.reshape(1,784)
        pixels_im2 = im2.reshape(1,784)
        pixels_im3 = im3.reshape(1,784)
        
        # Crear un gráfico de dispersión
        plt.figure(figsize=(8, 6))
        plt.scatter(range(1,785), pixels_im, s=5, c='orange', label = 'imagen 1')
        plt.scatter(range(1,785), pixels_im2, s=5, c='r', label = 'imagen 2')
        plt.scatter(range(1,785), pixels_im3, s=5, c='g', label = 'imagen 3')
        
        # Configurar etiquetas y título
        plt.xlabel('Número de pixel (de 1 a 784)')
        plt.ylabel('Intensidad del pixel en imagen 1, 2 y 3')
        plt.title('Comparación de intensidades en imagenes de la letra C')
        
        # Mostrar el gráfico
        plt.legend()
        plt.grid(True)
        plt.show()
