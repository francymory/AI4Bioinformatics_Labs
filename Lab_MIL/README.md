# Laboratorio sul MIL (Multiple-Instance Learning)

## MIL
Il Multi-Instance Learning (MIL) è una tecnica di apprendimento supervisionato debole in cui i dati di addestramento sono organizzati in bag (insiemi di istanze) e viene assegnata una label all'intero bag piuttosto che alle singole istanze. Questo è utile in scenari in cui l'annotazione dettagliata è costosa o impraticabile, come per l'ambito medico per l'identificazione di regioni patologiche senza annotazioni a livello locale.

I modelli MIL possono seguire due approcci principali:
1. Instance-Based Approach: Si assume che un bag classificato come positivo contenga almeno un'istanza positiva, mentre un bag negativo contenga tutte istanze negative. Per classificare il bag, si classificano prima tutte le istanze individualmente, poi il punteggio del bag è ottenuto tramite un'operazione di pooling (modelli: MaxPooling, MeanPooling).
2. Embedding-Based Approach: Le istanze vengono trasformate in uno spazio di embedding e poi aggregate (ad esempio con tecniche di attention o pooling) per formare una rappresentazione/embedding finale del bag che verrà poi classificato (modelli: DSMIL, BUFFERMIL, ABMIL, TRANSMIL).

In questo laboratorio ci concentriamo sulla Multiple-Instance Classification, ma esistono anche tecniche di Multiple-Instance Regression e Multiple-Instance Clustering.

## Parte 1 del Lab - MIL su MNIST

### Task 1: 
Run del codice che applica un semplice classificatore MIL al dataset MNIST, dove ogni bag contiene istanze (patches) multiple. Gli step applicati sono i seguenti: date le istanze di un bag, un *Feature extractor* pre-addestrato su MNIST estrae le loro features/embedding, che vengono poi aggregate in un singolo feature vector del bag tramite *Mean o Max pooling*. Il modello MIL, costituito da un *layer Fully_connected* e un *layer di Sigmoid activation*, prende in input gli embedding dei bag e produce una label binaria (0 o 1) su ogni bag.
   
### Task 2: 
Modifica del codice che aumenta il numero di istanze positive di un bag e testa le tecniche di mean e max pooling. Si analizza come questi cambiamenti influenzino la performance del modello (accuracy, precision, recall).

   Step:
   - Ho creato un nuovo dataset dove ho aumentato il numero di istanze positive in una bag a 7.
   - Ho mantenuto lo stesso Feature Extractor e MIL Classifier per classificazione binaria del task precedente.
   - Ho allenato e testato i 3 nuovi modelli, oltre al precedente sul vecchio dataset e Meanpooling: uno sul vecchio dataset e Maxpooling, uno sul nuovo dataset e Meanpooling, l'ultimo sul nuovo dataset e Maxpooling.
   
   Osservazioni:
   - Allenando e testando il modello sul vecchio dataset utilizzando il Maxpooling nel Feature Extractor, si nota un leggero miglioramento di accuracy e recall e un notevole miglioramento della precision.
   - Allendando e testando il modello sul nuovo dataset, sia con Meanpooling, sia con Maxpooling si verifica un notevole miglioramento della performance del modello in tutte e tre le metriche.

   
### Task 3:
Modifica del codice per supportare la Multi-Class classification, dove le labels dei bag vanno da 0 a 9.

   Step:
   - Ho scelto come regola di MIL Multi-Class classification la Majority-Rule: il bag prende la label della classe di patch più frequente nel bag. 
   - Ho quindi creato un dataset multiclasse, dove un bag ha più della metà di patch della label del bag stesso, e le restanti patch sono di classe randomica.
   - Ho modificato il MIL Classifier per renderlo multiclasse, sostituendo il Fully-connected layer che dall'embedding produceva un valore, con dei *layer lineari* che dato l'embedding del bag producono 10 valori (uno per ogni possibile classe). Ho inoltre sostituito la Sigmoid activation, con la *Softmax* usando la *CrossEntropy loss* nell'addestramento, che tramuta i 10 valori in un vettore di probabiltà, dove l'indice del vettore con la probabilità maggiore è la classe di predizione.
   - Ho quindi allenato e testato due classificatori multiclasse sul dataset multiclasse, uno che utilizza il Meanpooling per aggregare gli embedding delle feature, l'altro che sfrutta il Maxpooling.
  
   Osservazioni:
   - L'Accuracy del classificatore con Meanpooling è decisamente più elevata rispetto a quella calcolata per il classificatore con Maxpooling.

### Conclusioni:
- Classificazione Binaria MIL: si osserva che, aumentando il numero di istanze positive in un bag, diventa più semplice classificarlo come positivo. Inoltre, l'uso del Maxpooling si rivela più efficace per l'aggregazione delle feature, poiché si concentra sui valori più alti, catturando meglio la presenza di un'istanza positiva nel bag (ad esempio, un tumore).
- Classificazione Multi-Classe MIL con Majority-Rule: in questo caso, la tecnica di Maxpooling risulta meno efficace rispetto al Meanpooling. Questo perché, selezionando solo le feature con valore più elevato, Maxpooling enfatizza le istanze di classe con feature più forti, piuttosto che quelle più frequenti, portando a classificazioni meno accurate.
  
   
## Parte 2 del Lab - MIL sulle WSI (Camelyon16 dataset)

Le WSI (Whole Slide Images) sono immagini digitali ad alta risoluzione ottenute dalla scansione di interi vetrini istologici. Poichè sono troppo grandi per essere analizzate singolarmente, solitamente vengono utilizzate delle tecniche di pre-processing, che individuano le regioni significative di tessuto (in questo Lab è stato usato CLAM, che sfrutta Otsu per rimuovere il background), e da esse vengono estratte delle *patch*, che rappresentano porzioni specifiche del tessuto a una certa risoluzione.

In questo Lab per effettuare la classificazione MIL su WSI di Camelyon16, l'insieme delle patch (istanze) di una WSI (bag) viene prima processato da un feature extractor (DINO), gli embedding delle patch vengono poi aggregati con meccanismi di attention o pooling a seconda del modello, e il classificatore effettua la predizione finale sulla WSI (c'è o meno la metastasi). (In modo analogo a come avveniva per il dataset MNIST).

### Task 4: 
Testa i diversi modelli MIL, confrontandone struttura, funzionamento e performance.


#### Maxpooling e Meanpooling:
- Un Feature Extractor estrae gli embedding delle singole patch di una WSI, gli embedding vengono poi aggregati in un unico embedding della WSI tramite Maxpooling o Meanpooling.
  
#### DSMIL: 
- Modulo Dual-stream:
  1. Instance-level stream: Ogni patch della WSI viene trasformata in un embedding vettoriale usando un Feature Extractor pre-addestrato (CNN) con una loss contrastiva. Si classificano le singole patch dati i loro embedding e si applica MaxPooling per selezionare la patch più critica.
  2. Bag-level stream: Attraverso un meccanismo di self-attention si calcolano i pesi delle singole patch rispetto alla patch critica e si genera l'embedding finale della WSI, dove le patch più vicine alla critica avranno peso maggiore.
- L'Accuracy migliore di testing dopo 200 epoche di training è di: 0,86719

#### ABMIL:
- Ogni patch di una WSI viene trasformata in un embedding/vettore di features usando una CNN pre-addestrata. Le features estratte vengono proiettate in uno spazio latente tramite un MLP, dove viene calcolata l'importanza di ciascuna patch attraverso un meccanismo di attenzione. Viene calcolato l'embedding globale della WSI calcolando la somma pesata dall'attenzione delle feature delle patch.

#### TRANSMIL:
- Le patch della WSI vengono trasformate in embeddings attraverso una CNN pre-addestrata, agli embedding vengono aggiunte le informazioni spaziali delle patch nella WSI attraverso il Positional Encoding. Un Transformer Encoder modella le dipendenze tra tutte le patch tramite self-attention, permettendo di catturare relazioni tra patch distanti nella WSI. Il classification (CLS) token dell'Encoder contiene la rappresentazione globale della WSI.


### Task 5:
Visualizza gli attention weights a livello di patch del modello DSMIL.


