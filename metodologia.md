
# Metodologia

**Datasets**: Telco (net), Iris i Wine. Models: KNN (k=5), regressió logística
(max_iter=1000) i arbre de decisió (random_state=42).

**Partició**: 70 % entrenament / 30 % test, estratificada, amb 50 llavors (0-49).
Totes les llavors són compartides per tots els models i escalats.

**Escalat**: MinMaxScaler dins d'un Pipeline (s'ajusta només amb entrenament) i
versió sense escalar. Per Kmeans s'utlitzará StandardScaler.

**Experiments**: 

E1 50 particions amb escalat; 

E2 sense escalat; 

E3 Telcos ubmostrejat a 150, 500 i 2.000 files per poder comparar;

E4 (extra) K-Means amb ARI, NMI i accuracy aparellada.

**Mètriques**: accuracy, balanced accuracy, desviació típica, rang i IQR.

**Eines**: Python 3, pandas, scikit-learn, matplotlib.
