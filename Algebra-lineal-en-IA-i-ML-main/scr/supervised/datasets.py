# datasets.py
# Aquest fitxer només carrega dades. No entrena res.
# Sempre retorna el mateix format: X (taula per aprendre) i y (resposta).
# X = totes les columnes menys la resposta.
# y = només la columna resposta (la que volem endevinar).

import pandas as pd

#Funció per carregar el dataset Telco. Retorna X i y.
def carrega_telco():
    # 1. Llegim el csv net que ja está al repositori de Github.
    telco = pd.read_csv("./Data/processed/Telco_customers_processed.csv")

    # 2. Traiem la columna "Unnamed: 0" si hi és.
    # Aquesta columna és només el número de fila (0, 1, 2...).
    # No informa de res, i si la es deixa el model aprendria coses inutils.
    if "Unnamed: 0" in telco.columns:
        del telco["Unnamed: 0"]

    # 3. Separem X i y.
    # y és la columna que volem predir, Churn: 0 = El client no marxa, 1 = El client sí marxa.
    y = telco["Churn"]
    # X és tota la resta de columnes.
    #La funció drop() retorna un nou DataFrame sense la columna Churn, sense modificar el DataFrame original.
    X = telco.drop(columns=["Churn"])

    # 4. Retornem X i y  d'aquesta funció.
    return X, y

# Funció per carregar el dataset Iris. Retorna X i y.
def carrega_iris():
    # Iris ve amb sklearn, no cal csv.
    # 150 files, 4 columnes de números, 3 classes (50 files de cada), és un dataset molt perfecte.
    from sklearn.datasets import load_iris

    dades = load_iris()
    # dades.data és la taula X, dades.target és la resposta y.
    # Ho passem a DataFrame/Series perquè sigui igual que Telco.
    X = pd.DataFrame(dades.data, columns=dades.feature_names)
    y = pd.Series(dades.target)
    return X, y

#Funció per carregar el dataset Wine. Retorna X i y.
def carrega_wine():
    # Wine també ve amb sklearn.
    # 178 files, 13 columnes, 3 classes (59 / 71 / 48).
    from sklearn.datasets import load_wine

    dades = load_wine()
    X = pd.DataFrame(dades.data, columns=dades.feature_names)
    y = pd.Series(dades.target)
    return X, y
