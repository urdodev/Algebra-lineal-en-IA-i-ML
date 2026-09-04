'''
Script principal 1
Modelo: LogisticRegression
Metrics: ..
Antes de empezar, hay que aclarar que el procedimiento de estos script, por lo menos, las primeras partes, seran similares o iguales entre ellas, por lo que los comentarios solo se pondran una vez en las partes repetidas, en las nuevas si que se explicara el funcionamiento.
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

X = telco_csv.drop(columns=["Churn"])

#Definir la y, que sera la columna que se quiere predecir.

y = telco_csv["Churn"]

#Importar la funcion que sklearn que organizara el porcentaje de datos que se usaran para train y para test
from sklearn.model_selection import train_test_split

#Durante la practica se usara un 70% de los datos para train, y el otro 30% para test, random_state para poder reproducirlos.

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)


#Ahora toca normalizar, en los supervisados se usara MinMaxSclaer para escalar los datos entre el 0 y el 1
from sklearn.preprocessing import MinMaxScaler
#Se crea un objeto con la funcion para mayor facilidad segun la documnetacion.
scaler = MinMaxScaler()
#Se normalizan entre 0 y 1 las variables predictoras, ya que las y, que es churn ya esta entre 0 y 1.
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

#Se entrena el modelo con la funcion de Sklearn y se guarda en una variable.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#Se define el modelo como objeto dandole los parametros que necesitamos, en este caso, max_iter = 200, sigue siendo bajo para la cantidad de datos en el dataset, pero de igual manera se mantedra asi de momento.
log_customer = LogisticRegression(max_iter=200, random_state=42)
dec_tree_customer = DecisionTreeClassifier(random_state=42)

#se entrena con la X normalizada.
train_log_customer = log_customer.fit(X_train_norm, y_train)
train_tree_customer = dec_tree_customer.fit(X_train_norm, y_train)
#Predecir usando .predict(X_test_norm) guardando el resultado en una variable, predecir los resultados de X sin saber la respuesta correcta, solo usando X_test_nrm.

pred_log_customer = train_log_customer.predict(X_test_norm)
pred_tree_customer = train_tree_customer.predict(X_test_norm)

from sklearn.metrics import accuracy_score
#Calculamos el procentaje de aciertos.
accuracy_log = accuracy_score(y_test, pred_log_customer)
accuracy_tree = accuracy_score(y_test, pred_tree_customer)
print(f"Resultados predicciones LogisticRegression: {round(accuracy_log*100, 2)} %") 
print(f"Resultados predicciones DecisionTreeClassifier: {round(accuracy_tree*100, 2)} %") 
#Aun quedan 2 metricas mas, la matriz de confusión y la curva ROC.

#Matriz de confusión
'''
Para la matriz de confusión se crearan dos tablas en diferentes png, para asi poder compararlas con los otros modelos con mas facilidad.
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
#Definir el objeto Confusion Matrix con los datos correspondientes.
conf_mat_log = confusion_matrix(y_test, pred_log_customer)
conf_mat_tree = confusion_matrix(y_test, pred_tree_customer)

#Usar la funcion ConfusionMatrixDisplay para poder visualizarla como grafica.
disp_matrix_log = ConfusionMatrixDisplay(conf_mat_log, display_labels=["No Churn", "Churn"])
disp_matrix_log.plot()
#Titulo y carpeta donde guardar la grafica.
plt.title("Matriz de confusión - LogisticRegression")
plt.savefig("outputs/supervised/conf_matrix/matriz_confusion_logistic.png", dpi=150)

disp_matrix_tree = ConfusionMatrixDisplay(conf_mat_tree, display_labels=["No Churn", "Churn"])
disp_matrix_tree.plot()

plt.title("Matriz de confusión - DecisionTreeClassifier")
plt.savefig("outputs/supervised/conf_matrix/matriz_confusion_tree_classifier.png", dpi=150)


'''
Curva ROC
La ultima metrica de avaluación dentro de los modelos supervisados.
'''

from sklearn.metrics import roc_curve, RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, pred_log_customer, name="LogisticRegression")
plt.title("Corba ROC - LogisticRegression")
plt.savefig("outputs/supervised/roc_curve/roc_logistic.png", dpi=150)

RocCurveDisplay.from_predictions(y_test, pred_tree_customer, name="DecisionTreeClassifier")
plt.title("Corba ROC - DecisionTreeClassifier")
plt.savefig("outputs/supervised/roc_curve/roc_tree_classifier.png", dpi=150)