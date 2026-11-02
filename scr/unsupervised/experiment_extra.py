# experiment_extra.py - Experiment extra (no supervisat) - Kmeans
# Es fa amb els 3 datasets perquè així es pot comparar amb l'experiment 1.
# 3 datasets x 7 valors de k (2 a 8) x 5 llavors = 105 resultats.

import os
import sys
import pandas as pd

# datasets.py, que conté els datasets, és a la carpeta supervised, per això afegim aquesta ruta.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "supervised"))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from datasets import carrega_telco, carrega_iris, carrega_wine


# 1. Per què StandardScaler i no MinMax?
# K-Means calcula distàncies. Si una columna va de 0 a 7000 i una altra de 0 a 1,
# la gran mana i les altres gairebé no compten. Standard deixa totes les columnes
# amb mitjana 0 i desviació 1, així totes pesen el mateix.,


noms_datasets = ["Telco", "Iris", "Wine"]
valors_k = range(2, 9)
llavors = range(5)

files = []

for nom_dataset in noms_datasets:
    if nom_dataset == "Telco":
        X, y = carrega_telco()
    elif nom_dataset == "Iris":
        X, y = carrega_iris()
    else:
        X, y = carrega_wine()

    # Escalem tots els datasets de la mateixa manera.
    X_std = StandardScaler().fit_transform(X)

    for llavor in llavors:
        for k in valors_k:
            # n_init=10: prova 10 vegades on posar els centres i es queda la millor.
            # random_state=llavor per poder repetir el resultat.
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=llavor)
            grups = kmeans.fit_predict(X_std)

            # Silhouette: mesura com de separats i compactes són els grups (de -1 a 1).
            # Amb 7000 files calcular-ho sencer és molt lent, per això usem
            # una mostra fixa de 2000 files (la mateixa per a tots els k i llavors).
            sil = silhouette_score(X_std, grups, sample_size=2000, random_state=0)

            # ARI i NMI comparen els grups amb les classes reals (Churn o espècie).
            # Són les úniques mètriques que permeten comparar amb l'experiment 1.
            ari = adjusted_rand_score(y, grups)
            nmi = normalized_mutual_info_score(y, grups)

            files.append({
                "dataset": nom_dataset,
                "llavor": llavor,
                "k": k,
                "silhouette": sil,
                "inertia": kmeans.inertia_,
                "ARI": ari,
                "NMI": nmi,
            })

    print(f"{nom_dataset} fet: 7 valors de k x 5 llavors = 35 files.")

taula = pd.DataFrame(files)
os.makedirs("outputs/results", exist_ok=True)
taula.to_csv("outputs/results/EX.csv", index=False)
print(f"Guardat outputs/results/EX.csv amb {len(taula)} files.")

# Resum: mitjana de cada mètrica per dataset i k.
resum = taula.groupby(["dataset", "k"])[["silhouette", "ARI", "NMI"]].mean().round(4).reset_index()
resum.to_csv("outputs/results/EX_resum.csv", index=False)
print("Guardat outputs/results/EX_resum.csv.")
print()
print(resum.to_string(index=False))

