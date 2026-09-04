import pandas as pd
import numpy as np

# Primer, s'ha de llegir el dataset per poder veure una observació inicial, i veure que tot estigui al seu lloc.

Dades_brutes = pd.read_csv("Telco-costumers.csv")

# #Veure les 5 primeres files del dataset, com son moltes columnes per fila, només es mostren 5 columnes.
# print(Dades_brutes.head())
# print("="*35)
# #Veure les 5 últimes files del dataset.
# print(Dades_brutes.tail())
# print("="*35)
# #Et diu les dimensions del dataset
# print(Dades_brutes.shape)
# print("="*35)
# #Un resum del dataset per cada columna, especificant el tipus de dades de la columna.
# print(Dades_brutes.info())


# Aquí comencem amb el segon pas i l'últim abans de tractar aquestes dades, aquesta part es centrarà en la identificació de la quantitat de valors nuls o duplicats.

# print(Dades_brutes.isnull().sum())
# print("="*35)
# print(Dades_brutes.duplicated().sum())

# Hi ha sospitosament 0 nuls i 0 duplicats, això pot significar que els nuls no es representen com NaN, es poden representar com espais en blanc. Ara es comprovarà això: si una columna que té 0 valors nuls hauria de ser de tipus enter i en canvi és de tipus str o string (text), significa que molt probablement els valors nuls estiguin camuflats com a espais en blanc.
# print(Dades_brutes["TotalCharges"].dtype)

# Es confirma que és del tipus str. Un cop comprovat això, ens deixa pas a canviar el tipus de dades de la columna d'un str a un numèric. També s'haurà de tenir en compte que els espais en blanc no es poden transformar en un número, així que s'haurà d'especificar al programa que, quan doni error (que serà per aquests espais), els transformi en NaN (dades nul·les).

Dades_brutes["TotalCharges"] = pd.to_numeric(Dades_brutes["TotalCharges"], errors="coerce")
# print(Dades_brutes["TotalCharges"].isna().sum())
# Amb un print es pot observar que ara hi ha 11 valors nuls. Ara, abans de seguir, es farà el mateix amb totes les columnes amb la funció unique(), per detectar caràcters atípics a totes les columnes del dataset.

# llista_columnes_str = Dades_brutes.select_dtypes(str).columns.tolist()

# S'ha fet una llista de totes les columnes de tipus object per si torna a passar el mateix que amb l'anterior.

# for columna in llista_columnes_str:
#     print((columna, Dades_brutes[columna].unique()))

# S'ha confirmat que no hi ha més dades atípiques, l'única era "TotalCharges", que és la que la documentació assenyala com a tipus de dades erroni. Com que no hi ha més dades nul·les, es poden eliminar les onze files amb NaN a TotalCharges, ja que el dataset conté més de 7000 files, i 11 no afecten pràcticament de res.

Dades_brutes = Dades_brutes.dropna(subset=["TotalCharges"])

# print(Dades_brutes["TotalCharges"].isna().sum())
# Confirmació que s'han eliminat les files amb nuls

# El següent pas és treure columnes que no aporten res: primer les que només tenen un valor a totes les files (és a dir, que el valor és el mateix a totes les files d'aquesta columna), i després les que tenen el mateix nombre de valors únics que de files (és a dir, que cada fila té el seu propi valor), ja que aquesta tampoc aporta res.

# print(Dades_brutes.nunique())

# S'ha comprovat que no hi ha cap columna amb un únic valor repetit, però sí que n'hi ha una amb el mateix nombre de valors únics que de files: la de customerID. S'haurà de treure aquesta columna.

del Dades_brutes["customerID"]

# print(Dades_brutes.shape)

# Ja s'ha tret la columna i s'ha comprovat amb .shape que ara hi ha 20 columnes i no 21.

# Ara toca veure quines columnes influeixen directament en el "churn", que és el percentatge de persones que deixen el producte després d'un temps.


Contract_col = Dades_brutes.groupby("Contract")  # Agrupa les dades pels valors d'aquesta columna.
churn_grup = Contract_col["Churn"]
# Escull la columna Churn per a cada grup, així es pot fer la mitjana i veure si és rellevant o no.

percentatge_churn = churn_grup.apply(lambda x: (x == "Yes").mean())
# S'utilitza una funció lambda per calcular ràpidament la mitjana de "Yes" per grup de Contract en Churn. El resultat mostra que els grups són molt diferents entre si en aquesta columna, així que és molt important de cara al churn.

# Ara toca fer el mateix amb totes les columnes restants, es farà amb un bucle for, però primer s'ha d'agrupar en un subdataset les columnes de tipus str o object.

Col_influencia_churn = pd.DataFrame(Dades_brutes.select_dtypes(str))
# S'ha d'esborrar la columna Churn, ja que és la que es compararà amb la resta de columnes d'aquest dataset.

del Col_influencia_churn["Churn"]

for columna in Col_influencia_churn:
    group_column = Dades_brutes.groupby(columna)
    churn_grup = group_column["Churn"]
    percentatge_churn = churn_grup.apply(lambda x: (x == "Yes").mean())
    
# Després de revisar totes les columnes de tipus object, gender i PhoneService s'eliminaran perquè pràcticament no hi ha diferència entre les seves categories. Després m'he adonat que hi ha una categoria "No internet service" que és bastant rellevant, però es repeteix amb el mateix percentatge a moltes altres columnes; per no haver d'eliminar aquestes files, s'ha decidit fusionar aquesta categoria amb la categoria "No".

Dades_brutes["MultipleLines"] = Dades_brutes["MultipleLines"].replace("No phone service", "No")

Dades_brutes["StreamingTV"] = Dades_brutes["StreamingTV"].replace("No internet service", "No")

Dades_brutes["StreamingMovies"] = Dades_brutes["StreamingMovies"].replace("No internet service", "No")

Dades_brutes["OnlineSecurity"] = Dades_brutes["OnlineSecurity"].replace("No internet service", "No")

Dades_brutes["OnlineBackup"] = Dades_brutes["OnlineBackup"].replace("No internet service", "No")

Dades_brutes["DeviceProtection"] = Dades_brutes["DeviceProtection"].replace("No internet service", "No")

Dades_brutes["TechSupport"] = Dades_brutes["TechSupport"].replace("No internet service", "No")

# En aquestes columnes es repeteix el mateix patró, així que, per no perdre informació important, es fusiona amb "No". Es torna a fer el bucle per poder comprovar si ja s'han fusionat; ara ja es podrà decidir si són rellevants o si el marge de percentatge entre una categoria i l'altra és molt petit i no aporta res.
for columna in Col_influencia_churn:
    group_column = Dades_brutes.groupby(columna)
    churn_grup = group_column["Churn"]
    percentatge_churn = churn_grup.apply(lambda x: (x == "Yes").mean())
  #  print(percentatge_churn)

# S'ha posat el màxim de diferència en 5 punts; totes les columnes que no superin aquest límit s'eliminaran.
# Columnes que s'eliminaran: gender, PhoneService, MultipleLines, StreamingMovies i StreamingTV

del Dades_brutes["gender"]
del Dades_brutes["PhoneService"]
del Dades_brutes["MultipleLines"]
del Dades_brutes["StreamingMovies"]
del Dades_brutes["StreamingTV"]
# Confirmar l'eliminació de les columnes
# print(Dades_brutes.shape) #- Confirmat, queden 15 columnes que són rellevants
# Veure quantes columnes de tipus str (text) queden.
# print(Dades_brutes.dtypes)
# Comprovar que les columnes restants que tenien el "No internet service" fusionat amb el "No" efectivament tenen només 2 categories, "Yes" o "No".
# for columna in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]:
   # print(columna, " ", Dades_brutes[columna].nunique())

# Ara s'han de convertir totes les categories de les columnes en enters, ja que si no, a l'hora d'entrenar l'algorisme, no funcionarà. Les columnes amb dues categories, "Yes" o "No", formaran el grup 1, on el "Yes" es transformarà en 1 i el "No" en 0. Les que tenen més de dues categories faran servir una funció especial de pandas que serveix per dividir la columna original en tantes columnes noves com categories té la columna original, de manera que cada columna nova indiqui si la fila té o no aquella categoria original.

grup1 = ["Partner", "Dependents", "PaperlessBilling", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "Churn"]

grup2 = ["InternetService", "Contract", "PaymentMethod"]

# Transformació grup 1:
for columna in grup1:
    Dades_brutes[columna] = Dades_brutes[columna].replace("Yes", 1)
    Dades_brutes[columna] = Dades_brutes[columna].replace("No", 0)
# Aquí ocorre un problema, i és que pandas no canvia el tipus de dada de la columna; per això, tot i canviar les categories per enters, segueix sent de tipus object. Per això s'ha de forçar el tipus de dades de la columna amb .astype()

    Dades_brutes[columna] = Dades_brutes[columna].astype(int)
# Ara ja s'ha canviat el tipus de dada



# Transformació grup 2, funció get_dummies que transforma les categories de la columna en un dataset amb tantes columnes com categories:
new_internet = pd.get_dummies(Dades_brutes["InternetService"], prefix="Internet", prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=int)

# print (new_internet)

new_contract = pd.get_dummies(Dades_brutes["Contract"], prefix="Contract", prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=int)

# print (new_contract)

new_payment = pd.get_dummies(Dades_brutes["PaymentMethod"], prefix="Payment", prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=int)

# print (new_payment)

# Ara s'han de concatenar els 3 datasets nous amb Dades_brutes per, després, eliminar les columnes originals del dataset:

Dades_brutes = pd.concat([Dades_brutes, new_internet, new_contract, new_payment], axis=1)

# Revisar totes les columnes, per veure si s'ha concatenat correctament el dataset
# print(Dades_brutes.columns)
# print(Dades_brutes.shape)
# Després de confirmar que els datasets ja estan units, ara s'han d'eliminar les 3 columnes originals.
del Dades_brutes["PaymentMethod"]
del Dades_brutes["Contract"]
del Dades_brutes["InternetService"]

# Confirmar que ja estan eliminades i que totes les columnes del dataset siguin numèriques (int o float)
# print(Dades_brutes.shape)
# print(Dades_brutes.dtypes)

# Amb això es tanca la part de neteja de dades del .csv; ara s'exportarà i s'utilitzarà més endavant, ja net.

Dades_brutes.to_csv("Dades_netes_Telco_costumers.csv")
