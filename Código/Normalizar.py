# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :
Fecha : Febrero 2024
"""

import pandas as pd

redes = ["facebook","instagram","youtube","twitter","linkedin","flickr"]

def separar(dato):
        if dato is not None:
            datos_tupla = dato.strip(' //').split(' // ')
            datos_tupla = dato.split(' //')
            datos_tupla.pop()   
            return datos_tupla
        else:
            return []
    

def obtenerNombreRed(Url):
    for red in redes: 
        if red in Url:
            return red 
    if '@' in Url and 'gmail'  not in Url:
            return  'instagram'
        
def acomodar(df):
    columnas=['id_sede','Nombre','Url']
    res =pd.DataFrame(columns = columnas)
    res.columns =columnas
    for i in range(1,len(df)):
        if pd.notnull(df.iloc[i]['Url']) :
            urls =separar(df.iloc[i]['Url'])
            for j in range(0,len(urls)):
                url = urls[j]
                if obtenerNombreRed(url) != None:
                    nombre_red = obtenerNombreRed(url)
                    res = pd.concat([res, pd.DataFrame({'id_sede': [df.iloc[i]['id_sede']], 'Nombre': [nombre_red], 'Url': [url]})], ignore_index=True)     

    return res 

