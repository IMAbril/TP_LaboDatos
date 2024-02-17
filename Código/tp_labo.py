# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :
Fecha : Febrero 2024
"""
import pandas as pd

columnas_sede = ['id_sede', 'Nombre', 'Tipo', 'Estado', 'codigo_pais']
sedes = pd.DataFrame(columns=columnas_sede)

columnas_pais = ['Codigo','Pbi', 'Nombre']
paises = pd.DataFrame(columns = columnas_pais)