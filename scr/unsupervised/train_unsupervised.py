'''
Script principal 2
Modelos: PCA y KMeans
Metricas para Kmeans Silhoutte Score, porcentaje de Churn por cluster.
Metricas para PCA
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


#Ahora se usara StandardScaler para poder normalizar, ya que PCA calcula la varianza por columna para encontrar a las direcciones donde los datos varian mas, si no se normalizan los datos, afectara al calculo sobretodo las columnas con mas valores sin ser mas grandes.
from sklearn.preprocessing import StandardScaler
#Se crea un objeto con la funcion para mayor facilidad segun la documentación.
scaler = StandardScaler()


X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Importar los modelos no supervisados con sklearn.

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Se define el modelo como objeto dandole los parametros que necesitamos. En PCA, se usan 2 componentes principales para poder visualizar el resultado en un grafico de dispersion con mas facilidad
pca_customer = PCA(n_components=2, random_state=42)

#Se define le modelo Kmeans. Se usan 2 clusters porque el resultado con Churn, que tiene dos posibles respuestas (Si/No).
kmeans_customer = KMeans(n_clusters=2, random_state=42)

#se entrena con la X normalizada.

train_pca_customer = pca_customer.fit(X_train_std)
train_kmeans_customer = kmeans_customer.fit(X_train_std)

#PCA no tiene una funcion de predecir, ya que no lo hace como un modelo clasificador, pryecta los datos sobre los componentes que ha encontrado
X_train_pca = train_pca_customer.transform(X_train_std)
X_test_pca = train_pca_customer.transform(X_test_std)  

#Kmeans si tiene la función de prerdecirt, asigna cada fila al cluster mas cercano segun los otros que se han calculado en el entrenamiento.
cluster_train = train_kmeans_customer.predict(X_train_std)
cluster_test = train_kmeans_customer.predict(X_test_std)

#Se usaran metricas diferentes para poder comprar los modelso despues, ya que los modelos se evaluan de diferentes formas.
#PCA: varianza explciada. Muestra el porcentaje de valores originales que conservan los componenetes elegidos. Cuanto mas alto el %, menos informacion de se habra perdido.
varianza_explicada = train_pca_customer.explained_variance_ratio_


print(f"Varianza explicada: {round(varianza_explicada.sum()*100, 2)} %.")


#Para Kmeans, se evalua con silhouette Score, que mide la calidad de separacion de los clusters, va de -1 a 1, cuanto mas cerca de 1, mejor separados estan los clusters entre si y mas compactos.
from sklearn.metrics import silhouette_score

silhouette_train = silhouette_score(X_train_std, cluster_train)
silhouette_test = silhouette_score(X_test_std, cluster_test)

print(f"Silhouette Score: {round(silhouette_test, 3)}")

#La segunda metrica de Kmeans es el porcentae de Churn dentro de cada cluster. Permite poder comprobar si los grupos que ha encontrado el modelo separan a los clientes que se dan de baja o no. Esta metrica es la que permite poder comparar despues los 4 modelos entrenados en esta parte practica.
#Aun quedan 2 metricas mas, la matriz de confusión y la curva ROC.

resultados_test = pd.DataFrame({"cluster": cluster_test, "Churn": y_test.values})
 
porcentaje_churn_cluster = resultados_test.groupby("cluster")["Churn"].mean()
 
print("Porcentaje de Churn por cluster (test):")
print(round(porcentaje_churn_cluster * 100, 2))


'''
Graficas
'''
import os
import matplotlib.pyplot as plt

# Se usa un estilo mas moderno para que los gráficos se lean mejor.
plt.style.use("seaborn-v0_8-whitegrid")

os.makedirs("outputs/unsupervised/pca", exist_ok=True)
os.makedirs("outputs/unsupervised/kmeans", exist_ok=True)


def save_current_plot(path):
	plt.tight_layout()
	plt.savefig(path, dpi=150, bbox_inches="tight")
	plt.close()


def add_bar_labels(ax, fmt="{:.1f}%"):
	# Etiquetas numéricas encima de las barras para que no haya que estimar los valores a ojo.
	for container in ax.containers:
		ax.bar_label(container, labels=[fmt.format(value) for value in container.datavalues], padding=3)


# En modelos no supervisados no se usa matriz de confusión ni curva ROC directamente,
# porque esos gráficos necesitan valores que se han predecido en el test con los valores reales.
# Aquí se muestran gráficos que explican la estructura interna del modelo y cómo se relaciona con el campo Churn.

# Grafica 1: varianza explicada por componente principal. 
# Cada barra indica cuánta información del dataset conserva cada componente. Cuanto más alta sea, menos información se pierde al reducir dimensiones.
fig, ax = plt.subplots(figsize=(8, 4))
componentes = [f"CP{i + 1}" for i in range(len(varianza_explicada))]
valores_varianza = varianza_explicada * 100
ax.bar(componentes, valores_varianza, color="#4C78A8")
ax.set_title("Varianza explicada por PCA")
ax.set_xlabel("Componente principal")
ax.set_ylabel("Varianza explicada (%)")
ax.set_ylim(0, max(valores_varianza) * 1.25)
add_bar_labels(ax)
save_current_plot("outputs/unsupervised/pca/varianza_explicada_pca.png")


# Grafica 2: proyeccion de PCA coloreada por cluster
# Cada punto es un cliente. El color indica el cluster asignado por K-Means.
# Las X negras son los centros de cada grupo: cuanto más separados, mejor diferenciación visual.

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
	X_test_pca[:, 0],
	X_test_pca[:, 1],
	c=cluster_test,
	cmap="viridis",
	s=18,
	alpha=0.75,
)
centros_pca = pca_customer.transform(train_kmeans_customer.cluster_centers_)
ax.scatter(
	centros_pca[:, 0],
	centros_pca[:, 1],
	marker="X",
	s=180,
	c="#070707",
	edgecolors="white",
	linewidths=1.1,
)
ax.set_title("PCA con clusters de K-Means")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
colorbar = fig.colorbar(scatter, ax=ax, label="Cluster")
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(["Cluster 0", "Cluster 1"])

save_current_plot("outputs/unsupervised/pca/pca_clusters_test.png")


# Grafica 3: silhouette score 	
# Un valor más alto significa que los clusters están más compactos y mejor separados.
# Si se acerca a 1, la separación es buena; si baja de 0, los grupos se solapan mucho.

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["Train", "Test"], [silhouette_train, silhouette_test], color=["#59A14F", "#F28E2B"])
ax.set_title("Silhouette Score de K-Means")
ax.set_ylabel("Silhouette Score")
ax.set_ylim(-1, 1)
ax.axhline(0, color="#666666", linewidth=0.9)
add_bar_labels(ax, fmt="{:.3f}")
save_current_plot("outputs/unsupervised/kmeans/silhouette_score.png")
#En este caso estan muy separados y poco compactos, por lo que el modelo no ha encontrado una buena separacion entre los clusters,
#  aunque si que se puede ver que el cluster 1 tiene un porcentaje de churn mas alto que el cluster 0, 
# por lo que si que ha encontrado una  separacion entre los clientes que se dan de baja y los que no.


# Grafica 4: porcentaje de churn por cluster, 	La linea azul marca el churn medio del conjunto de test.
# Si un cluster queda por encima de esa línea, concentra más 'NO' que el promedio, este caso, al ser 2, el promedio esta hecho solo por esos 2 clusters.
fig, ax = plt.subplots(figsize=(7, 4))
churn_global = y_test.mean() * 100
porcentaje_churn_cluster.sort_index().mul(100).plot(kind="bar", ax=ax, color="#E15759")
ax.axhline(churn_global, color="#4C78A8", linestyle="--", linewidth=1.5, label=f"Churn medio test: {churn_global:.1f}%")
ax.set_title("Porcentaje de Churn por cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Churn (%)")
ax.set_ylim(0, 100)
add_bar_labels(ax)
ax.legend(loc="best")

save_current_plot("outputs/unsupervised/kmeans/churn_por_cluster.png")

