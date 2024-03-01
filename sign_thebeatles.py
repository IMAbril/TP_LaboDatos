#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024

Contenido: --

"""
#%% =============================================================================
# IMPORT módulos 
# =============================================================================
import pandas as pd
import visualizarImagenes as view #no se si lo vamos a usar pero lo pongo por las dudas

#from modulos import modulo as  -> Se importa el archivo modulo guardado en la carpeta modulos


#%% =============================================================================
# Cargamos los datos
# =============================================================================
carpeta = './datasets/'
sign_mnist = pd.read_csv(carpeta+'sign_mnist_train.csv')


#%% =============================================================================
# Funciones
# =============================================================================


#%% =============================================================================
# Código global (o fuera de las funciones)
# =============================================================================
