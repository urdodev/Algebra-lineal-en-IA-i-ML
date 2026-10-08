# TR – Àlgebra lineal en IA i ML (part pràctica)
Part pràctica del Treball de Recerca de Catalunya.
Autor: Rayan el Bakkali · Institut: INS Vinyes Velles

## Descripció
Es comparen tres models supervisats (KNN, regressió logística i arbre de decisió) sobre tres datasets (Telco Customer Churn, Iris i Wine). Es mesura la precisió (accuracy), com varia entre 50 particions entrenament/test, 
i l'efecte de l'escalat de dades, de la mida del dataset. Com a apartat extra, s'avalua K-Means amb mètriques externes i es discuteix per què la comparació amb els supervisats no és equitativa.

## Pregunta de recerca
Repetint l'entrenament amb diferents particions de dades, quin model supervisat és més precís a l'hora de predir cadascun dels tres datasets?
Quant varia el resultat entre particions? Afecten l'escalat i la mida del dataset?

## Estructura del repo

Data/ dades brutes (raw) i netes (processed) de Telco

scr/datacleaner/ neteja de Telco

scr/supervised/ experiment principal

scr/unsupervised/ K-Means

scr/test/ DummyClassifier

outputs/ resultats (CSV) i gràfiques

Part-teorica/ memòria del TR

metodologia.md metodologia del treball

Info-Dataset.md informació dels datasets

## Models
KNN: distància euclidiana entre vectors.


Regressió logística: producte escalar w·x + b i funció sigmoide.


Arbre de decisió: preguntes sobre una sola variable.
                

K-Means: distància al centroide.

## Mètriques
Accuracy score(principal), balanced accuracy, desviació típica entre particions. 
Extra: ARI, NMI i accuracy després d'aparellar clusters amb Kmeans.
