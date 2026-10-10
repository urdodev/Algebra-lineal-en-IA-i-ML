'''
Testeo del codigo de entrenamiento de la practica.
Modelo: DummyClassifier
Limitaciones: Sera un train test basico con un score al final.
'''

import pandas as pd
#Importar dataframe con .read_csv de la carpeta Data/processed.
telco_csv = pd.read_csv("./Data/processed/Telco_customers_processed.csv")
#Comprobacion rapida de que carga el csv correctamente
#print(telco_csv.head())


#Al imprimir el dataframe, se ve una columna Unnamed: 0, que es el indice de cada fila de datos. Es una columna inutil qeu no aporta nada, ya que es un distintivo por fila, por lo que nunca se repite, hay que eliminarla.

del telco_csv["Unnamed: 0"]

#Se ha eliminado la columna correctamente
#print(telco_csv.head())


#Definir la x, que seran los datos que usara el modelo para aprender patrones

X = telco_csv.drop(columns=["Churn"])

#Definir la y, que sera la columna que se quiere predecir.

y = telco_csv["Churn"]

#Importar la funcion que sklearn que organizara el porcentaje de datos que se usaran para train y para test
from sklearn.model_selection import train_test_split

#Durante la practica se usara un 70% de los datos para train, y el otro 30% para test, random_state para poder reproducirlos.

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

#Importar el modelo de skelarn.dummy

from sklearn.dummy import DummyClassifier

#Se usara en el parametro strategy "most_frequent", ya que siempre otorga el argumente mas frequente durante el train.
train_dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

#Predecir usando .predict(X_test) guardando el resultado en una variable, predecir los resultados de X sin saber la respuesta correcta, solo usando X_test.

pred_dummy = train_dummy.predict(X_test)

#Mostrar resultados con accuracy score usando y_test, que es la respuesta correcta, usando la funcion accuracy_score.

from sklearn.metrics import accuracy_score

#Usar accuracy_score para conseguir el procentaje de aciertos, en este caso, el porcentaje de aciertos sera el mismo que el procentaje de churn "No", ya que este modelo solo imita el valor más frecuente de Churn en el conjunto de entrenamiento (No), sin aprender ningún patrón real de los datos.
resul_dummy = accuracy_score(y_test, pred_dummy)

print(f"Resultados predicciones: {round(resul_dummy*100, 2)} %") 
#Output: Resultados predicciones: 73.41 %