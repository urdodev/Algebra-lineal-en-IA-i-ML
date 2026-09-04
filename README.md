# TR-AlgebraLineal-En-IA-y-ML
Este repo forma parte del Treball de Recerca de Catalunya por parte de un alumno.
- Hecho por: Rayan el Bakkali.
- Instituto: Ins Vinyes Velles

# Project-Info

Esta parte practica consiste en la limpieza de datos, entrenamiento y comparación entre diferentes modelos de Machine Learning desde un mismo dataset publico de Kaggle. 
La idea principal es coger un dataset sucio, con datos basura, nulos, etc. Para poder tratarlos y convertirlos en datos de mayor calidad, cosa que maximizara el resultado de los test con los modelos.
La segunda parte del proyecto consiste en el entrenamiento con este mismo dataset en 4 modelos diferentes: 2 supervisados y 2 no supervisados. Para luego visualizar los resultados en diferentes graficos, compararlos y comentar el porque de los resultados. Una de las mayores limitaciones sera el uso del modelo por refuerzo por su dificultad, como se ha acordado con el tutor.
En este repo se documentara todo el proceso, tanto el codigo, como la metodologia, errores y la comparación entre modelos.

# Objetivo del entrenamiento

Los modelos entrenados tendran que predecir la columna churn(Si/No), el principal problema aqui es que al no ser una columna numerica, los mdoelos supervisados de tipo regression no seran utiles, ya que se estara trabajando con un unico dataset para asi poder compararlos. El porque esta explicado en la memoria, por lo que se procedera con dos modelso de clasificacion dentro de los supervisados.

# Modelos Escogidos

Todos los modelos que se utilizaran en esta practica estan documentados y explicados en la parte teorica, asi se podra ver la logica que usa el modelo adaptada a codigo.

Los modelos supervisados elegidos son: Regresion Logistica y Arbol de decisiones.
    - Regresion logistica: Es el modelos mas relacionado con la algebra lineal de los explicados en la memoria, aqui es donde se vera la algebra lineal como la base de la IA.
    - Arbol de decisiones: A diferencia de la regresion logistica, este modelos no usa ninguna operacion con algebra lineal, por lo que me parece la mejor combinacion posible dentro de los supervisados


Los modelos no supervisados elegidos son: PCA y K-means.

    - PCA: Su funcionamiento se basa directamente en el calculo de vectores para reducir la dimensionalidad del dataset.
    - K-means:  A diferencia de PCA, K-means opera sobre la dsitancia de esos vectores, por lo que se complementa bien con PCA, ya que cada modelo opera de manera directa o indirecta sobre los vectores.


