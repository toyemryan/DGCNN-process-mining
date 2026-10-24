# Predizione della prossima attività con le Graph Neural Networks

> Progetto — *Big Data Analytics & Machine Learning*

Quattro reti neurali su grafi (**GraphSAGE**, **GATv2**, **GGNN** e **GCN**) imparano a predire **la prossima attività** di un processo aziendale in corso, a partire dall'inizio della sua esecuzione. Gli iperparametri vengono ottimizzati automaticamente con **Optuna**, e i quattro modelli vengono confrontati a parità di condizioni.

Questo repository riprende un codice esistente, sviluppato su macOS/Linux, lo **adatta a Windows** con una GPU NVIDIA e ne **modifica lo scheletro** (sezione 3).

<!-- Da completare: riferimento all'articolo / al codice originale -->

---

## 1. Far funzionare il progetto su Windows

### Prerequisiti

- Python 3.10 o 3.11 (testato con **3.11.9**)
- Una scheda NVIDIA con un driver recente (testato con una **RTX 4070 Laptop**, CUDA 12.8)

### Installazione

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

In `requirements.txt`, commentare le due righe della versione CPU:

```
# torch==2.3.1
# torch-geometric==2.3.1
```

Poi installare le dipendenze e PyTorch con CUDA:

```powershell
pip install -r requirements.txt
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
```

Verificare che la GPU venga riconosciuta (il risultato deve contenere `True`):

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Modifiche apportate per Windows

| # | Problema | Soluzione |
|---|---|---|
| 1 | Il programma termina senza fare nulla: i dati sono forniti come `.g.zip` e non vengono rilevati | Decomprimere gli archivi in `dataset/` (comando qui sotto) |
| 2 | La RAM si satura (16 GB) e poi crash con `EOFError: Ran out of input`: su Windows ogni processo parallelo ricarica tutto Python e copia i dati | `config.py`: `MAX_WORKERS = 2` invece di `cpu_count() - 3` |
| 3 | Il `requirements.txt` installa PyTorch **CPU**: la GPU non viene usata | Installazione di PyTorch 2.8 + CUDA 12.8 (vedi sopra) |
| 4 | Crash alla ripresa dopo un'interruzione: lo stato `COMPLETED` non esiste in Optuna | `COMPLETE` + conteggio dei trial `PRUNED` |
| 5 | Gli errori non indicavano la loro origine | `traceback.print_exc()` invece di `print(e)` |

Decompressione dei dati:

```powershell
Get-ChildItem dataset\*.zip | ForEach-Object { Expand-Archive $_.FullName -DestinationPath dataset -Force }
```

---

## 2. Esecuzione

Scegliere il modello da addestrare in `hyper_parameters_search.py`, lasciando attiva **una sola** riga di import:

```python
from gnn_SAGE import SAGE_GNN as GNN
#from gnn_GATv2 import GATv2_GNN as GNN
#from gnn_GGNN import GGNN_GNN as GNN
#from gnn_GCN import GCN_GNN as GNN
#from gnn_GCN_Base import GCN_GNN as GNN
```

Salvare il file, poi lanciare:

```powershell
python -u hyper_parameters_search.py 2>&1 | Tee-Object log.txt
```

Il messaggio rosso `NativeCommandError` mostrato da PowerShell non è grave: proviene dalla barra di avanzamento.

I risultati vengono scritti in `models/` e gli studi Optuna in `various/optuna/`. Un dataset già completato viene saltato all'esecuzione successiva. **Prima di addestrare un altro modello**, rinominare le due cartelle, altrimenti il nuovo addestramento verrebbe saltato o mescolato con il precedente:

```powershell
Rename-Item models models_sage
Rename-Item various\optuna optuna_sage
```

Convenzione usata: `models_sage`, `models_gatv2`, `models_ggnn`, `models_gcn`.

---

## 3. Modifiche al codice originale

Tutti i modelli confrontati condividono lo stesso protocollo di addestramento e lo stesso **scheletro**, ottenuto modificando quello del codice originale:

| # | Limite del codice originale | Modifica | Dove |
|---|---|---|---|
| 1 | I dati di addestramento non vengono mai mescolati: ogni batch contiene i prefissi delle stesse poche tracce, sempre nello stesso ordine | `shuffle=True` nel DataLoader di addestramento (validazione e test invariati) | `hyper_parameters_search.py` |
| 2 | Il readout (somma, media o massimo) mescola tutti i nodi: il modello non sa quale sia l'attività in corso, l'informazione più utile per predire la successiva | Il vettore dell'**ultimo nodo** del prefisso viene concatenato al readout. Il target non è mai visibile: si aggiunge solo l'ultima attività già eseguita | file dei modelli |
| 3 | Fino a 6 layer senza normalizzazione né connessioni residue: ad ogni layer il vettore di un nodo viene sostituito, e la sua identità si perde | **LayerNorm** dopo ogni layer e **connessione residua** dal 2° layer: ogni layer aggiunge informazione invece di sostituirla | file dei modelli |

Il pruner di Optuna usa inoltre `n_warmup_steps=10` (invece di 30), per interrompere prima i trial poco promettenti.

Le modifiche 2 e 3 non fanno parte dei layer forniti da PyTorch Geometric: sono implementate nello scheletro, in modo che i modelli differiscano solo per il tipo di layer. La loro utilità è verificata nella sezione 7 (*Excursus: GCN_Base*).

---

## 4. Come funziona

```
Event log  →  Instance Graphs  →  Prefix-IGs  →  GNN  →  Prossima attività
```

1. **I dati**: ogni esecuzione di un processo (una *traccia*) è rappresentata da un grafo orientato, l'**Instance Graph**. I nodi sono le attività, descritte dalla loro codifica one-hot e da 3 caratteristiche temporali; gli archi sono le loro dipendenze.
2. **I prefissi**: ogni grafo viene suddiviso in grafi parziali (le prime 2, 3, 4… attività). Ognuno è associato all'attività che segue realmente.
3. **Il modello**: da 2 a 6 layer GNN, un *readout* (somma, media o massimo) che riassume il grafo, a cui si aggiunge il vettore dell'ultima attività, poi due layer lineari che predicono la prossima attività (sezione 5).
4. **L'ottimizzazione**: **Optuna** prova 20 combinazioni di iperparametri per dataset. Ogni combinazione viene addestrata fino a 150 epoch, con early stopping dopo 20 epoch senza miglioramento. Il modello migliore viene scelto in base alla loss di validazione, poi valutato sul test set.

| File | Ruolo |
|---|---|
| `config.py` | Parametri globali |
| `data_encoding.py` | Generazione dei prefix-IGs |
| `gnn_SAGE.py` | GraphSAGE |
| `gnn_GATv2.py` | GATv2: teste di attenzione scelte da Optuna |
| `gnn_GGNN.py` | GGNN: GRU, aggregazione e ingresso scelti da Optuna |
| `gnn_GCN.py` | GCN |
| `gnn_GCN_Base.py` | GCN con lo scheletro originale, senza le modifiche 2 e 3 (sezione 7, *Excursus*) |
| `hyper_parameters_search.py` | Ricerca degli iperparametri, addestramento e test |
| `prova_predizione.py` | Prova del modello addestrato su un prefisso (sezione 6) |
| `migliori_iperparametri.py` | Metriche e iperparametri scelti da Optuna per ogni modello (sezione 7) |

---

## 5. Architetture utilizzate

### Cosa succede quando si dà un prefisso al modello

Esempio: una richiesta è in corso e sono già avvenute due attività. Quale sarà la prossima?

```
Evento A  ──►  Evento B          →  prossima attività?
```

1. **Le attività si scambiano informazioni (layer GNN, da 2 a 6 volte).** Ogni attività è descritta da una scheda di 32 numeri (quale attività è e quando è avvenuta). Ogni attività legge la scheda di quelle che la precedono e aggiorna la propria: alla fine, la scheda dell'*Evento B* dice «sono l'Evento B, e prima di me c'è stato l'Evento A».
2. **Si fa un riassunto (readout).** Le schede di tutte le attività vengono combinate in **una sola** (somma, media o massimo), che descrive l'intero prefisso. Accanto si aggiunge la scheda dell'**ultima attività**, la più utile.
3. **Si decide (piccola rete finale con 2 layer lineari).** La rete dà un punteggio a **ogni attività possibile**, trasformato in probabilità, ad esempio *Evento C* 80 %, *Evento D* 15 %, *Evento E* 5 %. L'attività più probabile è la predizione.

Durante l'addestramento si confronta la predizione con l'attività reale: se è sbagliata, il modello corregge leggermente i suoi pesi.

I quattro modelli differiscono solo per il **passo 1**, cioè per il modo in cui un'attività combina la propria scheda con quelle dei predecessori.

### GraphSAGE (`gnn_SAGE.py`)

Ogni attività combina la propria scheda con la **media** di quelle dei predecessori, con due matrici di pesi distinte:

```
nuovo(i) = W_sé · x(i)  +  W_vicini · media( x(j) )     j = predecessori di i
```

Le due matrici sono apprese separatamente: il modello può trattare in modo diverso «chi sono» (`W_sé`) e «cosa è successo prima di me» (`W_vicini`).

### GATv2 (`gnn_GATv2.py`)

Invece di una media semplice, ogni attività **impara quanto pesare ciascun predecessore** (meccanismo di *attenzione*, pesi che sommano a 1):

```
nuovo(i) = Σ α(i,j) · W · x(j)                         j = predecessori di i, e i stesso
α(i,j)   = softmax_j( a · LeakyReLU( W_a · [x(i) ‖ x(j)] ) )
```

I pesi α vengono calcolati confrontando ogni coppia (attività, predecessore): ad esempio, con un solo predecessore, `nuovo(i) = 0,7 · W·x(i) + 0,3 · W·x(pred)`, dove 0,7 e 0,3 sono decisi dal modello. Rispetto al file di partenza è stato corretto un errore: `aggr='mean'` divideva una seconda volta i pesi di attenzione, che quindi non sommavano più a 1.

Ogni **testa** di attenzione calcola il proprio modo di pesare i predecessori. Il numero di teste e il modo di combinarle sono scelti da Optuna:

| `concat` | Combinazione | Dimensione di ogni testa | Parametri* |
|---|---|---|---:|
| `False` | **media** delle teste | `layers_size` | ≈ 276 000 × numero di teste |
| `True` | teste **messe in fila** | `layers_size ÷ num_heads` | ≈ 276 000, qualunque sia il numero di teste |

`num_heads` ∈ {1, 2, 4}: il valore 3 è escluso perché con `concat=True` non divide le dimensioni esplorate (es. 160 ÷ 3 non è intero).

### GGNN (`gnn_GGNN.py`)

Ad ogni passo di propagazione, ogni attività riceve i messaggi dei suoi predecessori e aggiorna la propria scheda con una **GRU**, che decide valore per valore quanto conservare della scheda precedente e quanto prendere della nuova informazione. È il layer usato da SR-GNN (Wu et al., AAAI 2019) per predire il prossimo elemento di una sessione, un problema molto simile al nostro.

```
messaggio(i) = Σ W · h(j)                              j = predecessori di i (somma, media o massimo)
nuovo(i)     = (1 − z) ⊙ h(i)  +  z ⊙ candidato(i)     z ∈ [0,1], calcolato dalla GRU
```

`candidato(i)` è il nuovo vettore proposto a partire dal messaggio. Per ogni valore del vettore, la porta `z` dosa vecchio e nuovo: con `h(i) = [0, 0, 1]`, `candidato = [0,2, 0,9, 0,1]` e `z = [0, 0,8, 0,1]` si ottiene `nuovo(i) = [0, 0,72, 0,91]`: il terzo valore (l'identità dell'attività) viene quasi conservato, il secondo prende l'informazione del predecessore.

Un solo layer `GatedGraphConv` con `hidden_layers + 2` passi (stesso numero degli altri modelli). Poiché tutti i passi avvengono all'interno di un unico layer, la **LayerNorm** è applicata una sola volta, dopo la propagazione, e la **connessione residua** non è aggiunta: la GRU svolge già lo stesso ruolo. Due scelte lasciate a Optuna:

| Iperparametro | Opzioni | Significato |
|---|---|---|
| `ggnn_aggr` | `add`, `mean`, `max` | Come si combinano i messaggi dei predecessori |
| `ggnn_input` | `pad`, `linear` | Le 32 schede in ingresso completate con zeri, oppure trasformate da un layer appreso |

### GCN (`gnn_GCN.py`)

Ogni attività fa una **media pesata** della propria scheda e di quelle dei predecessori, poi applica **una sola matrice di pesi**: a differenza di GraphSAGE, non distingue «chi sono» da «cosa è successo prima di me». I pesi della media sono fissati dal numero di collegamenti di ogni attività, una normalizzazione pensata per grafi **non orientati**:

```
nuovo(i) = W · Σ x(j) / √(grado(i) · grado(j))          j = predecessori di i, e i stesso
```

Su una catena, con un solo predecessore: `nuovo(i) = W · (0,50 · x(i) + 0,71 · x(pred))`. L'attività e il predecessore vengono mescolati **prima** della matrice `W`, con pesi non appresi, e il predecessore conta più dell'attività stessa.

Optuna sceglie anche `improved`, che raddoppia il peso del legame di ogni nodo con sé stesso: `nuovo(i) = W · (0,67 · x(i) + 0,41 · x(pred))`, e l'attività conta più del suo predecessore. In PyG 2.8 questo parametro viene ignorato se non si passa `edge_weight`: il modello passa quindi un peso pari a 1 per ogni arco, in modo che `improved` abbia effetto.

### Confronto

| Modello | Come combina i predecessori | Parametri* |
|---|---|---:|
| GraphSAGE | Media, due matrici (sé / predecessori) | ≈ 273 000 |
| GATv2 | Attenzione, teste scelte da Optuna | ≈ 276 000 – 928 000 |
| GGNN | Somma / media / massimo + GRU | ≈ 339 000 – 344 000 |
| GCN | Media pesata, una sola matrice | ≈ 166 000 |

\* Esempio con 3 layer nascosti da 160 neuroni, sul dataset Receipt (32 valori in ingresso, 29 attività possibili).

### Iperparametri cercati da Optuna

Comuni a tutti i modelli: `learning_rate` (1e-5 – 1e-2), `batch_size` (128, 192, 256), `hidden_layers` (0 – 4), `layers_size` (64 – 256), `dropout` (0 – 0,4), `readout` (somma, media, massimo). Quelli specifici vengono aggiunti solo per il modello interessato, in `get_hyperparameters`:

```python
if GNN.__name__ == 'GATv2_GNN':
    parameters['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
    parameters['concat'] = trial.suggest_categorical('concat', [True, False])
if GNN.__name__ == 'GCN_GNN':
    parameters['improved'] = trial.suggest_categorical('improved', [True, False])
if GNN.__name__ == 'SAGE_GNN':
    parameters['aggregator'] = trial.suggest_categorical('aggregator', ['mean'])
if GNN.__name__ == 'GGNN_GNN':
    parameters['ggnn_aggr'] = trial.suggest_categorical('ggnn_aggr', ['add', 'mean', 'max'])
    parameters['ggnn_input'] = trial.suggest_categorical('ggnn_input', ['pad', 'linear'])
```

---

## 6. Provare il modello su un prefisso

Lo script **`prova_predizione.py`** carica il modello migliore già addestrato, gli dà un prefisso e mostra le attività successive più probabili, accanto all'attività reale.

```powershell
python prova_predizione.py
```

Esempio con GraphSAGE su Receipt (prefisso di 3 nodi della traccia `case-10810`):

```
Prefisso             : artificial_start -> Confirmationofreceipt -> T02Checkconfirmationofreceipt
Attività reale       : T04Determineconfirmationofreceipt (insieme: train)
Il modello propone   :
   T04Determineconfirmationofreceipt              83.6 %
   T06Determinenecessityofstopadvice              15.4 %
   T03Adjustconfirmationofreceipt                  0.9 %
```

Il modello dà l'83,6 % all'attività corretta, ma lascia il 15,4 % a T06: ha imparato che dopo T02 esistono due percorsi possibili, e quale dei due è il più frequente.

### Da dove vengono i prefissi

```
<Dataset>_processed.g            →   data_encoding.py   →   dataset/<Dataset>_tensors/*.pt
tutti i grafi completi,              (fase prefix-IG)        un file per prefisso
uno per traccia
```

- Il file **`.g`** contiene **tutte le tracce**, una dopo l'altra: per ogni traccia, le righe `v` (le attività, cioè i nodi) e le righe `e` (gli archi tra le attività). Ogni traccia è identificata dal suo `case_id`.
- Ogni traccia viene tagliata in **prefissi** (le prime 2, 3, 4… attività), ognuno salvato in un file **`.pt`** con l'attività successiva come risposta da indovinare. Per Receipt: 1 434 tracce → 7 143 prefissi.
- Il nome di un file dice tutto: `Receipt_case-10810_3.pt` = dataset Receipt, traccia `case-10810`, prefisso dei primi 3 nodi.
- Ogni traccia inizia con `artificial_start` e finisce con `artificial_end`: sono marcatori di inizio e fine, non attività reali. `artificial_end` non è mai una risposta da indovinare.

### Cosa fa lo script

1. **Carica il prefisso** (`.pt`): le attività in forma numerica (`x`), gli archi (`edge_index`) e la risposta corretta.
2. **Ricostruisce i nomi delle attività** a partire dalla codifica one-hot (`<Dataset>_ohe_activity.json`).
3. **Carica il modello migliore**: costruisce un modello vuoto con gli stessi iperparametri, poi vi carica i pesi appresi (`load_state_dict`).
4. **Calcola la predizione**: un punteggio per ogni attività possibile, trasformato in probabilità (`softmax`).
5. **Mostra le 3 attività più probabili** accanto a quella reale.

### Per provare altri casi

In cima allo script:

| Parametro | Esempio | Ruolo |
|---|---|---|
| `DATASET` | `'Receipt'` | Il dataset |
| `PREFISSO` | `'case-10810_3'` | La traccia e la lunghezza del prefisso (un file di `dataset/<DATASET>_tensors`) |
| `MODELLI` | `'models_sage'` | La cartella dei risultati dell'addestramento |

Per elencare i prefissi disponibili di una traccia:

```powershell
Get-ChildItem dataset\Receipt_tensors -Filter "Receipt_case-10810_*" -Name
```

Per un altro modello, l'import in cima allo script e `MODELLI` devono corrispondere, ad esempio per GATv2: `from gnn_GATv2 import GATv2_GNN as SAGE_GNN` e `MODELLI = 'models_gatv2'`.

> Lo script indica l'insieme del prefisso (`train`, `validation` o `test`). Per una prova onesta, scegliere un prefisso del **test set**, che il modello non ha mai visto durante l'addestramento.

---

## 7. Risultati

### In sintesi

- **I quattro modelli sono molto vicini tra loro**: a parità di scheletro, la scelta del layer cambia poco i risultati. Sui dataset BPI gli scarti sono ≤ 0,003 di val loss, dello stesso ordine del caso.
- **GGNN è il più regolare** (rango medio 1,8 su 4) e ha la migliore accuracy su 3 dataset. **GraphSAGE** vince su Receipt e Helpdesk.
- **Receipt è l'unico dataset con scarti netti**: GraphSAGE e GGNN sono davanti a GATv2 e GCN.
- **Lo scheletro conta più del layer**: senza le modifiche 2 e 3, lo stesso GCN perde fino a 0,09 di val loss (*Excursus: GCN_Base*).
- **Helpdesk resta bloccato** a ≈ 30 % di F1 macro con tutti i modelli: è un limite dei dati (prefissi identici portano ad attività diverse), non dell'architettura.

### Metriche

Modello migliore di ogni dataset (scelto sulla loss di validazione), valutato sul test set. In **grassetto** il migliore di ogni riga.

**Val loss** (più bassa = meglio)

| Dataset | SAGE | GATv2 | GGNN | GCN |
|---|---:|---:|---:|---:|
| Receipt | **0,4433** | 0,4539 | 0,4480 | 0,4597 |
| Helpdesk_no_resources | **0,6624** | 0,6708 | 0,6628 | 0,6683 |
| BPI20R | 0,3894 | 0,3908 | **0,3884** | 0,3910 |
| BPI20P | 0,3633 | 0,3653 | **0,3624** | 0,3636 |
| BPI12WC | 0,4833 | 0,4826 | 0,4828 | **0,4825** |

**Accuracy (%)**

| Dataset | SAGE | GATv2 | GGNN | GCN |
|---|---:|---:|---:|---:|
| Receipt | 86,66 | 86,24 | **86,70** | 86,58 |
| Helpdesk_no_resources | **79,16** | 78,97 | 78,67 | 78,55 |
| BPI20R | 85,43 | 85,45 | **85,51** | 85,36 |
| BPI20P | 85,48 | 85,44 | **85,87** | 85,57 |
| BPI12WC | 78,06 | **78,33** | 78,20 | 78,24 |

**F1 macro (%)** — tutte le attività con lo stesso peso, comprese le rare

| Dataset | SAGE | GATv2 | GGNN | GCN |
|---|---:|---:|---:|---:|
| Receipt | **52,76** | 45,15 | 51,48 | 48,75 |
| Helpdesk_no_resources | **30,50** | 30,36 | 30,16 | 30,04 |
| BPI20R | 40,52 | 42,59 | **42,60** | 42,54 |
| BPI20P | 46,56 | 47,35 | **48,14** | 47,51 |
| BPI12WC | 61,97 | **64,05** | 63,31 | 63,55 |

**F1 weighted (%)** — attività pesate per frequenza

| Dataset | SAGE | GATv2 | GGNN | GCN |
|---|---:|---:|---:|---:|
| Receipt | 84,90 | 84,22 | **85,09** | 84,70 |
| Helpdesk_no_resources | **74,52** | 74,25 | 73,91 | 73,69 |
| BPI20R | 80,76 | 80,71 | **80,85** | 80,72 |
| BPI20P | 81,97 | 82,63 | **83,03** | 82,71 |
| BPI12WC | 75,83 | **76,35** | 75,91 | 75,95 |

**Rango in val loss**

| Modello | Receipt | Helpdesk | BPI20R | BPI20P | BPI12WC | Rango medio |
|---|---:|---:|---:|---:|---:|---:|
| GGNN | 2 | 2 | 1 | 1 | 3 | **1,8** |
| GraphSAGE | 1 | 1 | 2 | 2 | 4 | 2,0 |
| GCN | 4 | 3 | 4 | 3 | 1 | 3,0 |
| GATv2 | 3 | 4 | 3 | 4 | 2 | 3,2 |

### Interpretazione

- **GraphSAGE**: migliore in val loss e in F1 macro su Receipt e Helpdesk (52,8 % su Receipt): tratta meglio le attività rare.
- **GGNN**: il modello più costante, migliore in val loss su BPI20R e BPI20P e in accuracy su Receipt, BPI20R e BPI20P.
- **GATv2**: l'attenzione non porta un guadagno chiaro. Su Receipt ogni attività ha un solo predecessore, quindi l'attenzione sceglie solo tra l'attività stessa e il suo predecessore. GATv2 è invece il migliore su BPI12WC in accuracy, F1 macro e F1 weighted.
- **GCN**: primo in val loss su BPI12WC e vicino agli altri sui dataset BPI; l'unico scarto netto è su Receipt (+0,016 di val loss, F1 macro 48,8 % contro 52,8 % di GraphSAGE).

### Perché queste differenze

*Ipotesi coerenti con i risultati.*

- **I quattro modelli sono vicini** perché gli Instance Graph sono quasi delle catene (in Receipt ogni attività ha un solo predecessore). Con un solo predecessore, media (SAGE), attenzione (GATv2), somma (GGNN) e media pesata (GCN) fanno quasi la stessa cosa: combinare l'attività con quella precedente. Inoltre i modelli condividono lo stesso scheletro, che pesa molto nella decisione, e sono tutti vicini al limite dei dati.
- **GGNN è il più regolare** grazie alla GRU: invece di sommare sempre per intero vecchio vettore e correzione (come la connessione residua), decide valore per valore quanto conservare e quanto aggiornare, e può anche dimenticare. Applicata ad ogni passo, legge la traccia attività dopo attività come una rete ricorrente, il modello classico per la predizione della prossima attività.
- **GCN resta indietro su Receipt** perché usa una sola matrice: l'attività e il suo predecessore vengono mescolati prima di essere trasformati, con pesi fissati dalla normalizzazione, e il modello non può più distinguerli. Su Receipt, che ha molte attività rare, confondere l'identità delle attività costa caro. Coerentemente, su Receipt Optuna sceglie `improved=True`, che dà più peso all'attività stessa.

### Excursus: GCN_Base e l'utilità delle modifiche

Lo stesso GCN, addestrato con lo stesso protocollo ma con lo **scheletro originale** (`gnn_GCN_Base.py`):

| Dataset | Val loss GCN_Base | Val loss GCN | Guadagno | F1 macro GCN_Base → GCN |
|---|---:|---:|---:|---:|
| Receipt | 0,5509 | **0,4597** | −0,091 | 41,99 → 48,75 |
| Helpdesk_no_resources | 0,6803 | **0,6683** | −0,012 | 29,56 → 30,04 |
| BPI20R | 0,4030 | **0,3910** | −0,012 | 42,05 → 42,54 |
| BPI20P | 0,4151 | **0,3636** | −0,052 | 45,57 → 47,51 |
| BPI12WC | 0,4943 | **0,4825** | −0,012 | 60,35 → 63,55 |

GCN_Base era ultimo ovunque; con le modifiche, GCN raggiunge gli altri modelli: **lo scarto veniva dallo scheletro, non dal layer**.

1. **Ultimo nodo**: senza, il readout mescola tutte le attività e il modello non vede direttamente l'attività in corso.
2. **Connessione residua**: senza, ad ogni layer il vettore viene sostituito da un mescolamento con i predecessori, e l'identità dell'attività si perde. Optuna sceglieva infatti 0 layer nascosti per GCN_Base su Receipt; con la connessione residua, GCN sfrutta 3 layer nascosti.
3. **LayerNorm**: mantiene i vettori sulla stessa scala da un layer all'altro, rendendo l'addestramento più stabile.

**Iperparametri di GCN_Base**: `readout = max` su tutti e 5 i dataset, dropout molto basso (≤ 0,08), learning rate spesso più alto (fino a 3,6e-3), 0 layer nascosti su Receipt e BPI12WC e 3 su BPI20R e BPI20P. Le scelte di `improved` (3 `False`, 2 `True`) sono casuali, perché il parametro non aveva effetto.

### Limite dei dati

**Un paradosso: stesso passato, futuro diverso.** Due prefissi di Helpdesk con la stessa sequenza di attività portano, nella realtà, ad attività diverse:

| Prefisso | Sequenza | Attività reale | GGNN: s6 / s9 |
|---|---|---|---|
| `s1937_3` (validation) | `artificial_start → s1 → s8` | **s6** | 75,9 % / 20,6 % |
| `s3789_3` (test) | `artificial_start → s1 → s8` | **s9** | 76,7 % / 19,0 % |

Il modello dà quasi le stesse probabilità ai due prefissi (anche GraphSAGE: 72,7 % / 23,0 %) e predice `s6` in entrambi i casi: giusto per il primo, sbagliato per il secondo. Nei dati, delle **3 055 tracce** che iniziano con `s1 → s8`, il **76,8 %** prosegue con `s6` e il **20,1 %** con `s9`. Due architetture diverse ritrovano esattamente queste frequenze: sono **ben calibrate**, ma nessuna può superare il 76,8 % su questo prefisso, perché nulla nei dati distingue i due casi.

**Cosa descrive un evento.** Nel file `.g`, ogni attività (riga `v`) contiene:

| Informazione | Usata dal modello |
|---|---|
| Nome dell'attività (codifica one-hot) | ✅ |
| `norm_time`: posizione nel calendario del log | ✅ |
| `trace_time`: tempo dall'inizio della traccia | ✅ |
| `prev_event_time`: tempo dall'attività precedente | ✅ |
| `start_time`, `end_time` (orari reali) | ❌ |
| `org_resource` (chi ha svolto l'attività) | ❌ (presente in 4 dataset su 5, assente in Helpdesk) |

Per Helpdesk queste informazioni sono poco utili: solo **9 attività**, generiche e anonimizzate (`s1` … `s9`), nessuna risorsa, e tempi poco informativi. Per il prefisso `s1 → s8`, `norm_time` vale 0,53 ± 0,27 per le tracce che vanno verso `s6` e 0,55 ± 0,27 per quelle che vanno verso `s9` (indistinguibili), mentre `trace_time` e `prev_event_time` valgono 0, perché sono calcolati dall'**inizio** dell'attività: la durata dell'ultima attività non è mai visibile.

**Perché Helpdesk è il dataset più difficile.** Una semplice tabella di frequenze basata sulla sequenza di attività raggiunge il **78,7 %**, e i modelli il **79,2 %**: hanno già raggiunto il limite. Le caratteristiche temporali non aggiungono quasi nulla, perché `norm_time` misura la data e non l'avanzamento del ticket, e le durate utili valgono spesso zero. La debolezza di Helpdesk viene quindi dalla **povertà dell'informazione disponibile** (9 attività generiche, nessuna risorsa, tempi poco informativi), e non dai modelli.

### Iperparametri scelti da Optuna

Elenco completo per modello e dataset: `python migliori_iperparametri.py` (tabella anche in `migliori_iperparametri.csv`).

**Scelte specifiche**

| Dataset | GATv2: teste / concat | GGNN: aggregazione / ingresso | GCN: improved |
|---|---|---|---|
| Receipt | 1 / False | mean / pad | True |
| Helpdesk_no_resources | 2 / True | add / linear | False |
| BPI20R | 2 / True | max / pad | False |
| BPI20P | 2 / False | mean / linear | False |
| BPI12WC | 1 / False | mean / linear | True |

**Tendenze comuni** (20 modelli migliori: 4 modelli × 5 dataset)

| Iperparametro | Scelta prevalente | Cosa significa |
|---|---|---|
| `num_heads` (GATv2) | 1 o 2, **mai 4** | Bastano poche teste: i nodi hanno pochi predecessori tra cui scegliere |
| `readout` | `max` (10/20) o `add` (7/20), `mean` solo 3/20 | Conta l'attività più marcata o il numero di attività, non la loro media; BPI20P e BPI20R preferiscono `add` (3/4 ciascuno) |
| `batch_size` | 256 (13/20) | I batch più grandi sono preferiti |
| `learning_rate` | tra 1e-4 e 1e-3 (15/20) | Zona di apprendimento stabile |
| `hidden_layers` | GraphSAGE: sempre 3 o 4; GATv2 e GGNN: 0 o 1 in 7 casi su 10; GCN: da 0 a 3 | GraphSAGE sfrutta reti più profonde; GATv2 e GGNN preferiscono reti poco profonde |
| `layers_size` | 160 – 224 (14/20) | Dimensione media dello spazio esplorato |

**Cosa significa**: le stesse combinazioni ritornano spesso tra modelli diversi. Con `seed=0`, i primi trial di Optuna sono estratti a caso ma sempre allo stesso modo: il trial 0 (`lr 4,4e-4`, batch 256, 3 layer da 160, dropout 0,17, `max`) è il migliore 6 volte su 20. Con 20 trial la ricerca esplora solo una parte dello spazio, e gli scarti tra combinazioni sono piccoli: non esiste un ottimo netto, ma una **zona di buoni iperparametri**. I modelli sono poco sensibili alla scelta precisa al suo interno.

> **Limiti del confronto**: ogni modello è stato addestrato una sola volta, con un solo seed. Un secondo addestramento di GGNN, identico al primo, ha dato scarti fino a 0,0015 di val loss e 0,4 punti di accuracy: differenze di questo ordine sono dovute al caso. GATv2 e GGNN hanno 2 iperparametri in più di GraphSAGE, GCN uno (`improved`). I risultati di GCN_Base provengono da un addestramento separato, con lo stesso protocollo.

---

## AUTORI

- TOYEM TEZEM RYAN PARFAIT
- MICHELE MARZIONI
- GIACOMO PRINCIPI
