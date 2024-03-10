#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024
Contenido : En las siguientes lineas se encuentra el proceso llevado a cabo y documentado en el informe.
            Cada proceso hace uso de imports donde se encuentran las funciones detalladas. Decidimos separar para una mejor comprensión.
"""
#%% =============================================================================
# IMPORT módulos 
# =============================================================================
import pandas as pd
import KNN
import exploracion 
#%% =============================================================================
# Cargamos los datos
# =============================================================================
carpeta = '/Users/Roju2/OneDrive/Desktop/tp2/'
sign_mnist = pd.read_csv(carpeta+'sign_mnist_train.csv')


#%% =============================================================================
#Dividimos nuestro código en secciones de trabajo.

#%%==============================================================================
#ANALISIS EXPLORATORIO 
#================================================================================

#Comenzamos por conocer la cantidad de muestras de cada letra

exploracion.cantidad_De_Muestras() #Ejecutar para conocer la cantidad de muestras por letra

#%%
#Luego comenzamos Por explorar las variaciones imagen a imagen

exploracion.imagenes_Letra_C() #Muestra las imagenes a comparar

exploracion.comparar_Intensidades_LetraC() #Compara y grafica las intensidades de las imagenes
#%%
#Por último, comparamos como se comportan las componentes de letras  similares y distintas.
#Para ello redujimos la dimensión de los datos

exploracion.imagenes_Letras_PCA() #Muestra las letras a comparar

exploracion.comparar_E_M() #Compara dos letras similares

exploracion.comparar_E_L() #Compara dos letras distintas


#%%=============================================================================
#CLASIFICACIÓN BINARIA 
#===============================================================================
#Nos propusimos crear un modelo que clasificara si una letra era la letra A o la letra L

#Primero nos fijamos la proporción de muestras de cada letra

KNN.muestras() #Imprime las cantidad de muestras
#%%

#Luego miramos cual sería el numero de atributos adecuado y cual el de K
#%%
#Creamos y entrenamos el modelo con k=5 y le pasamos distintos numeros de atributos
KNN.clasificador_atributosVariables() #Muestra el grafico que representa lo obtenido e imprime 

#%%
#Probamos como varia el rendimiento de acuerdo a tomar conjuntos aleatorios de 3 atributos
KNN.clasificador_3Atributos_Variables()#Muestra gráfico obtenido y una tabla con la informacion del print

#%%
#Probamos como varía el rendimiento de acuerdo al K elegido  
KNN.clasificador_K_Variable() #Muestra grafico obtenido e imprime información

