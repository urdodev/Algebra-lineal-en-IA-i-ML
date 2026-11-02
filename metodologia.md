CAPÍTOL 4: METODOLOGIA

Aquest capítol descriu el material estudiat, el disseny de l’estudi amb el seu pla de treball, i els mètodes de recollida d’informació amb els procediments i les tècniques emprades. L’estudi compara models d’inteligència artificial (algorismes qeu aprenen patrons a partir de dades) en tres experiments que segueixen sempre la mateixa base i esquelet, de manera que els resultats puguin ser comaprables entre ells.


3.4.1. MATERIAL ANALITZAT I DOMINI DE L’ESTUDI

El material analitzat són tres conjunts de dades (datasets) públics i molt utilitzats en recerca o aprenentatge autodidacta, escollits perque representa tres situacions molt diferents:

Telco (IBM, tret de Kaggle): 7043 clients d’una companyia de telecomunicacions descrits originalment 21 variables (mesos contractats, factuaració, serves… Barreja tipus de dades numeriques, binaries i categoriques), que després de la neteja de dades, eliminanant algunes columnes i altres dividirles, es mantenen el mateix numero de variables, 21, peró ara 19 son binaries i 2 numeriques.L’objectiu és predir la variable Churn (si el client es dona de baixa del servei). És el dataset més realista; presenta un desequilibri de classes notable (nomès un 26% de casos positius de Churn), i, que després de la neteja de files amb dades incompletes es queden 7032.

Iris (scikit-learn): 150 flors de 3 espècies (50 de cada una), amb 4 variables numeriques en centimetres (llargàries i amplades de pètals i sèpals ). Totes les varaibles tenen escalaes gairebe iguals i les classes son equiliibrades.

Wine (scikit-learn): 178 vins de 3 varietats (59, 71 i 48), amb 13 variables químiques (alcojol, magnesi,...), les escales de les quals son molt desiguals; algunes es mouen en decimes i altres en milers.





3.4.2. DISSENY DE L’ESTUDI I PLA DE TREBALL

El disseny és experimental i comparatiu: es plantegen quatre preguntes de recerca (P1 - P4), es dissenyen tres experiments per contestar-les, i cada hipòtesi inicial es valora amb els resultats obtinguts.

Experiment 1 (E1) - contesta P1 i P2
Tres models supervisats: KNN (k = 5), Regressió Logística (max_iter = 1000) i Arbre de decisió (random_state = 42).
S’entrenen sobre els 3 datasets, repetint l’entrenament 50 vegades amb llavors diferents (el nombre aleatori que defineix la partició de les dades), amb les dades escalades de 0 a 1 amb MinMaxScaler. P1 pregunta quin model és més precís i P2 quant varia el resultat entre particions.

Experiment 2 (E2) -  contesta P3
Identic a E1 però sense cap escalat. Com cada llavor comparteix la mateixa partició en E1 i E2, es pot comparar la diferencia de resultats per cada model i dataset.

Experiment 3 (E3) -  contesta P4
Es reproduira amb un model no supervisat, K-Means, que agrupa les dades sense veure mai les etiquetes (la resposta només es fa servir per valorar el resultat). S’entrena amb StandardScaler(mitjana 0 i desviació 1) perque les seves operacions es basen en distancies; es prova desde k = 2 fins a  k = 8 amb 5 llavors. 
Total del càlcul: 3 datasets x 7 valors de k x 5 llavors = 105 entrenaments.


PLA DE TREBALL 
Neteja de les dades de Telco (optimització de variables i eliminació de files incompletes).
Creació d’un mòdul comú per carregar els conjunts de dades, que retorna sempre el mateix format.
Codificació dels 3 experiments, cada un exporta un CSV dins del repositori de github.
Execució dels entrenaments i generació dels gràfics i taules resum (Figures).
Anàlisi i comentament dels resultats pregunta per pregunta, comparant els models entre si i evaluar cada hipòtesi incial.

Es mantindrà un control de versions amb Github amb diversos commits durant tot el procès


3.4.3. MÈTODES DE RECOLLIDA D’INFORMACIÓ, PROCEDIMENTS I TÈCNIQUES

La informació no es recull amb enquestes ni mostres pròpies, degut a la complexitat d’aquestes ultimes, la informació prové dels datasets públics descrits a l’apartat 3.1. El que realment es recull són els resultats dels entrenaments, i per aconseguir-ho es fa servir les seguents métriques:

Métriques dels experiments supervisats: l’exactitud (% de files encertades), la desviació típica (quant varien els resultats al voltant de la mitjana) i el rang (màxim i mínim entre les 50 llavors).
Métriques de l’experiment no supervisat: ARI (Adjusted Rand Index) compara els grups (k) obtinguts amb les classes reals mirant l’atzar (0 = atzar, 1 = perfectament exacte).

A cada llavor es divideix el dataset en un 70% per entrenar i un 30% per a la prova (test);  el model apren a l’entrenament i s’avalua al test amb dades que no ha vist mai.

Comparació emparellada entre E1 i E2: La mateixa partició s’entrena amb i sense escalat i es resten les dues exactituds mitjanes, una diferencia positiva vol dir un millor resultat amb escalat.

