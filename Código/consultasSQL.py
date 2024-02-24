# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del Grupo: The Beatles 
Fecha : Febrero 2024
"""
import pandas as pd
import DataFrames as data
from inline_sql import sql, sql_val 

sedes = data.sedes_df

redesSociales = data.redSocial_df

secciones = data.secciones_df

paises = data.paises_df


# Punto 1
# Tabla con: País, Sedes, secciones promedio y PBI per Capita 2022

NomPBI_df = sql ^ """
                    SELECT Nombre, PBI, Codigo AS codigo_pais
                    FROM paises AS Nom_PBI
                    """

cantSecciones_df = sql ^ """
                        SELECT id_sede, count(*) AS Cant_secciones, codigo_pais
                        FROM sedes AS s_count
                        GROUP BY id_sede, codigo_pais
                        """

promSecciones_df = sql ^ """
                        SELECT codigo_pais, AVG(Cant_secciones) AS prom_secciones
                        FROM cantSecciones_df
                        GROUP BY codigo_pais
                        """

cantSedes_df = sql ^ """
                        SELECT codigo_pais, count(*) AS cant_sedes
                        FROM cantSecciones_df
                        GROUP BY codigo_pais
                        """


tablaFinal_df = sql ^ """
                        SELECT NomPBI_df.Nombre AS País, 
                                cantSedes_df.cant_sedes AS Sedes, 
                                promSecciones_df.prom_secciones AS "secciones promedio", 
                                NomPBI_df.PBI AS "PBI per Cápita 2022 (U$S)"
                        FROM NomPBI_df
                        JOIN cantSedes_df ON NomPBI_df.codigo_pais = cantSedes_df.codigo_pais
                        JOIN promSecciones_df ON cantSedes_df.codigo_pais = promSecciones_df.codigo_pais
                        ORDER BY cant_sedes DESC, País ASC
                        """


##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##
# Punto 2
# Tabla con: Región geografica, Países con sedes Argentinas y Promedio PBI per Capita 2022

region_df = sql ^ """
                  SELECT Region, count(Region) AS Cant
                  FROM paises
                  GROUP BY Region
                  """


promPBI_df = sql ^ """
                    SELECT Region, AVG(CAST(REPLACE(PBI, ',', '.') AS DECIMAL(9,2))) AS PBI
                    FROM paises
                    GROUP BY Region
                  """

regionPBI_df = sql ^ """
                    SELECT promPBI_df.Region AS "Región geográfica", 
                           region_df.Cant AS "Países Con Sedes Argentinas",
                           promPBI_df.PBI AS "Promedio PBI per Cápita 2022 (U$S)"
                    FROM promPBI_df
                    JOIN region_df ON promPBI_df.Region = region_df.Region
                    ORDER BY promPBI_df.PBI DESC
                  """


##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##
# Punto 3
# Buscamos cuál es la vía de comunicación de las sedes en cada país

urlJoin_df = sql ^ """
                  SELECT sedes.Nombre,
                      sedes.id_sede AS Sede,
                      sedes.codigo_pais,
                      redesSociales.Nombre AS "Red Social"
                  FROM sedes
                  JOIN redesSociales ON sedes.URL_Red_Social = redesSociales.URL
                  """


getRedSocial_df = sql ^ """
                          SELECT urlJoin_df.codigo_pais, urlJoin_df."Red Social"
                          FROM urlJoin_df
                          GROUP BY "Red Social", codigo_pais
                          """

cantRedesPais_df = sql ^ """
                      SELECT getRedSocial_df.codigo_pais, 
                             count(getRedSocial_df."Red Social")
                      FROM getRedSocial_df
                      GROUP BY codigo_pais
                      """


##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##
# Punto 4
# Reporte con la información de redes sociales


getInfoRedes_df = sql ^ """
                      SELECT paises_df.Nombre, 
                             sedes.id_sede, 
                             sedes.URL_Red_Social AS URL
                      FROM paises_df
                      JOIN sedes ON paises_df.codigo = sedes.codigo_pais
                      """

infoRedes_df = sql ^ """
                      SELECT getInfoRedes_df.Nombre, 
                             getInfoRedes_df.id_sede AS Sede, 
                             redesSociales.Nombre AS "Red Social",
                             getInfoRedes_df.URL,                                
                      FROM getInfoRedes_df
                      JOIN redesSociales ON getInfoRedes_df.URL = redesSociales.URL
                      ORDER BY getInfoRedes_df.Nombre, Sede, "Red Social", getInfoRedes_df.URL
                      """

