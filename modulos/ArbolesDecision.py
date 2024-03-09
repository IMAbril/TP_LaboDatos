#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:14:21 2024

@author: francisco
"""

#%%
# Entrenar el modelo de árbol de decisión con la mejor profundidad
best_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=10)
best_clf.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión en el conjunto de prueba: {test_accuracy}")

# Generar métricas de clasificación multiclase
print(classification_report(y_test, y_pred))

# Generar matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# #%%
# # Visualización de la distribución de clases
# plt.figure(figsize=(8, 6))
# data_vocales['label'].value_counts().plot(kind='bar')
# plt.title('Distribución de Clases')
# plt.xlabel('Clase')
# plt.ylabel('Frecuencia')
# plt.show()

# # Visualización de la precisión media en función de la profundidad del árbol
# plt.figure(figsize=(10, 6))
# plt.plot(depths, mean_scores, marker='o', color='b')
# plt.title('Precisión Media de Validación Cruzada vs. Profundidad del Árbol')
# plt.xlabel('Profundidad del Árbol')
# plt.ylabel('Precisión Media')
# plt.xticks(depths)
# plt.grid(True)
# plt.show()

# # Visualización de la matriz de confusión
# """
# # Entrenar el modelo de árbol de decisión con la mejor profundidad
# # Generar matriz de confusión
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Visualizar la matriz de confusión
# plt.figure(figsize=(8, 6))
# plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Matriz de Confusión')
# plt.colorbar()
# classes = [str(i) for i in range(len(np.unique(y)))]
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes)
# plt.yticks(tick_marks, classes)
# plt.xlabel('Clase Predicha')
# plt.ylabel('Clase Verdadera')

# # Anotar los valores en la matriz
# thresh = conf_matrix.max() / 2.
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         plt.text(j, i, format(conf_matrix[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if conf_matrix[i, j] > thresh else "black")

# plt.tight_layout()
# plt.show()
# """

