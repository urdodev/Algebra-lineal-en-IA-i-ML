# Metodologia

El dataset utilizado es Telco Customer Churn (Kaggle), limpiado previamente en la fase de limpieza (Neteja_dades_Telco.py).

La variable que se predecirá es Churn (0 = No, 1 = Yes).

**Entrenamiento**
Los datos se dividen en entrenamiento (70%) y  test (30%) con train_test_split.

**Escaladores**
Se aplican dos escaladores distintos, dependiendo  del modelo:

MinMaxScaler para Regresión Logística y Árbol de Decisión.
StandardScaler para PCA y K-Means.


**Métricas de evaluación**
Supervisados: accuracy score, matriz de confusión, curva ROC.

No supervisados: silhouette score (calidad de separación de los clústers) y, despues, porcentaje de Churn dentro de cada clúster encontrado por K-Means.

**Herramientas**
Se usó Python V3, conjuntamente con las librerias Pandas para limpieza del dataset, Scikit-Learn para entrenamineto del modelo y evaluación del resultado, y matplotlib para visualizar resultados con graficas.