# Laboratorio sul MIL (Multiple-Instance Learning)

## MIL
Il Multi-Instance Learning (MIL) è una tecnica di apprendimento supervisionato debole in cui i dati di addestramento sono organizzati in bag (insiemi di istanze) e viene assegnata una label all'intero bag piuttosto che alle singole istanze. Questo è utile in scenari in cui l'annotazione dettagliata è costosa o impraticabile, come per l'ambito medico per l'identificazione di regioni patologiche senza annotazioni a livello locale.

I modelli MIL possono seguire due approcci principali:
1. Instance-Based Approach: Si assume che un bag classificato come positivo contenga almeno un'istanza positiva, mentre un bag negativo contenga tutte istanze negative. Per classificare il bag, si classificano prima tutte le istanze individualmente, poi il punteggio del bag è ottenuto tramite un'operazione di pooling (modelli: MaxPooling, MeanPooling).
2. Embedding-Based Approach: Le istanze vengono trasformate in uno spazio di embedding e poi aggregate (ad esempio con tecniche di attention) per formare una rappresentazione/embedding finale del bag che verrà poi classificato (modelli: DSMIL, BUFFERMIL, ABMIL, TRANSMIL).

In questo laboratorio ci concentriamo sulla Multiple-Instance Classification, ma esistono anche tecniche di Multiple-Instance Regression e Multiple-Instance Clustering.

## Parte 1 del Lab - MIL su MNIST

1. Task 1: Run del codice che applica un semplice classificatore MIL al dataset MNIST, dove ogni bag contiene istanze (patches) multiple. Gli step applicati sono i seguenti: date le istanze di un bag, un *feature extractor* pre-addestrato su MNIST estrae le loro features/embedding, che vengono poi aggregate in un singolo feature vector del bag tramite *mean o max pooling*. Il modello MIL, costituito da un *layer fully_connected* e un *layer di sigmoid activation*, prende in input gli embedding dei bag e produce una label binaria (0 o 1) su ogni bag.
2. Task 2: Modifica del codice che aumenta il numero di istanze positive di un bag e testa le tecniche di mean e max pooling. Si analizza come questi cambiamenti influenzino la performance del modello (accuracy, precision, recall)
3. Task 3: Modifica del codice per supportare la Multi_class classification, dove le labels dei bag vanno da 0 a 9. 




