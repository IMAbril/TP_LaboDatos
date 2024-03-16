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
from modulos import KNN_correccion as KNN, exploracion, ArbolesDecision_corregido as Arbol
#%% =============================================================================
# Cargamos los datos
# =============================================================================
carpeta = '/Users/Roju2/OneDrive/Desktop/'
sign = pd.read_csv(carpeta +'sign_mnist_train.csv')


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

#Vemos el comportamiento de ambas clases analizando sus componentes principales
KNN.comparar_A_L()


#%%
#Luego nos propusimos investigar que conjuntos  de atributos eran mejores para evaluar los modelos
#Para ello tuvimos en cuenta la variabilidad pixel a pixel
#consideramos 3 atributos

KNN.KNN_3Atributos_mayorVariabilidad() #Imprime score obtenido

KNN.KNN_3Atributos_variabilidadMedia() #imprime score obtenido

KNN.KNN_3Atributos_menorVariabilidad() #imprime score obtenido

#%%
#Despues hicimos lo mismo, pero variando la cantidad de atributos

KNN.KNN_AtributosVariables_mayorVariabilidad()

KNN.KNN_AtributosVariables_variabilidadMedia()

KNN.KNN_AtributosVariables_menorVariabilidad()

#A continuacion se muestra un grafico de lo obtenido
KNN.grafico_atributos_variables()

#%%
#Luego, variamos los valores de k, utilizando tres atributos

KNN.KNN_k_variabilidadMayor()

KNN.KNN_k_variabilidadMedia()

KNN.KNN_k_variabilidadMenor()

#A continuacion se muestra un grafico de lo obtenido
KNN.grafico_k_variable()

#%%
#Por ultimo, realizamos cross validation para elegir el mejor modelo

KNN.cross_validation()
#%%=============================================================================
#CLASIFICACIÓN MULTICLASE
#===============================================================================

#Desarrollamos un modelo que pudiera clasificar una imagen de acuerdo a que vocal es.
#Para ello, implementamos un arbol de decision

#Primero comenzamos conociendo la distribución de las clases.
Arbol.vocales()

#%%

#Luego construimos modelos y evaluamos cual sería una profundidad adecuada
Arbol.profundidad_arbol()
#se imprimen las profundidades, score y una matriz de confusion
#Además se obtiene el gráfico de la Matriz de confusion

