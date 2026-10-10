# experiment.py - Experiment 1
# Pregunta P1 i P2: quin model és més precís a cada dataset? Quant varia entre particions?
# 3 datasets x 3 models x 50 particions cada model= 450 notes o resultats.
# Cada nota és un accuracy al test. Al final es guarda en un CSV i es mirará la mitjana.

import os
import pandas as pd

# 1. Eines de sklearn que es farà servir.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# 2. Importem les funcions com a moduls per carregar dades.
# datasets.py ha d'estar a la mateixa carpeta que aquest fitxer.
from datasets import carrega_telco, carrega_iris, carrega_wine


# 3. Petita ajuda: donat un nom, crea el model nou.
# Ho fem amb if perquè sigui fàcil de llegir.
# KNN k=5: mira els 5 veïns més propers (distància euclidiana).
# LogReg max_iter=1000: 1000 perquè convergeixi fins i tot sense escalar.
# Arbre random_state=42 sempre mateixa aleatorietat.
def crea_model(nom):
    if nom == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if nom == "LogReg":
        return LogisticRegression(max_iter=1000)
    # Si no és cap dels dos, és l'arbre.
    return DecisionTreeClassifier(random_state=42)


# 4. Llistas dels datasets y els models que provarem.
noms_datasets = ["Telco", "Iris", "Wine"]
noms_models = ["KNN", "LogReg", "Tree"]
llavors = range(50)  # 50 particions.

# 5. Aquí guardarem totes les files, una per cada prova.
files = []

# 6. Bucle principal: per cada dataset farà 3 models x 50 particions = 150 files.
for nom_dataset in noms_datasets:
    # Carreguem X i y segons el dataset.
    if nom_dataset == "Telco":
        X, y = carrega_telco()
    elif nom_dataset == "Iris":
        X, y = carrega_iris()
    else:
        X, y = carrega_wine()

    # Per cada llavor (cada partició diferent) farà:
    for llavor in llavors:
        #  70% train / 30% test.
        # stratify=y manté el % de classes a train i test.
        # Per exemple, si Telco té 26% churn, el test també tindrà ~26%.
        # random_state=llavor que fa que tothom pugui repetir la mateixa partició.
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=llavor)

        # Per cada model (KNN, LogReg, Tree) farà:
        for nom_model in noms_models:
            # El scaler s'ajusta només amb train (fit a train).
            # Així el test no es filtrarà.
            tub = Pipeline([
                ("scaler", MinMaxScaler()),
                ("model", crea_model(nom_model)),])

            # Entrenem només amb train.
            tub.fit(X_train, y_train)

            # Prediccion sense veure la resposta del test (y_test).
            pred = tub.predict(X_test)

            # Calculem les 2 mètriques.
            # accuracy: encerts / total. És la principal metrica que mesura el rendiment del model.
            acc = accuracy_score(y_test, pred)
            # balanced: mitjana d'encerts per classe. Quan hi ha desequilibri entre les classes es utilitza per evitar porblemas externs.
            bal = balanced_accuracy_score(y_test, pred)

            # Guardem la fila amb un diccionari.
            files.append({
                "dataset": nom_dataset,
                "llavor": llavor,
                "model": nom_model,
                "accuracy": acc,
                "balanced": bal,
            })

    # Comprobació rapida que s'han fet les 150 files per aquest dataset.
    print(f"{nom_dataset} fet: 3 models x 50 llavors = 150 files.")

# 7. Passem la llista a taula i la guardem en un fitxer CSV.
# Per facilitar l'anàlisi.
taula = pd.DataFrame(files)
#Funció per crear la carpeta si no existeix. exist_ok=True evita error si ja existeix.
os.makedirs("outputs/results", exist_ok=True)
taula.to_csv("outputs/results/E1.csv", index=False)
print(f"Guardat outputs/results/E1.csv amb {len(taula)} files.")

# 8. Fem la taula resum per comparar (P1 i P2).
# Una fila per dataset i model, amb mitjana, desviació, mínim, màxim, rang i IQR.
resum = taula.groupby(["dataset", "model"])["accuracy"].agg(["mean", "std", "min", "max"])
resum["range"] = resum["max"] - resum["min"]
q = taula.groupby(["dataset", "model"])["accuracy"].quantile([0.25, 0.75]).unstack()
resum["IQR"] = q[0.75] - q[0.25]
resum = resum.round(4).reset_index()
resum.to_csv("outputs/results/E1_resum.csv", index=False)
print("Guardat outputs/results/E1_resum.csv per comparar.")

