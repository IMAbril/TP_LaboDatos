# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :
Fecha : Febrero 2024
"""
import pandas as pd

columnas_sede = ['id_sede', 'Nombre', 'Tipo', 'Estado', 'codigo_pais','Red-Social']
sedes = pd.DataFrame(columns=columnas_sede)

columnas_pais = ['Codigo','PBI', 'Nombre','Region', 'Nivel-Ingreso']
paises = pd.DataFrame(columns = columnas_pais)

columnas_seccion = ['id_sede','Descripcion']
secciones = pd.DataFrame(columns=columnas_seccion)
