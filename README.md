# Predizione della prossima attività con le Graph Neural Networks

> Progetto — *Big Data Analytics & Machine Learning*

Una rete neurale su grafi (**GraphSAGE**) impara a predire **la prossima attività** di un processo aziendale in corso, a partire dall'inizio della sua esecuzione. Gli iperparametri vengono ottimizzati automaticamente con **Optuna**.

Questo repository riprende un codice esistente, sviluppato su macOS/Linux, e lo **adatta a Windows** con una GPU NVIDIA.

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

> Le correzioni 4 e 5, così come `n_warmup_steps=10` per il pruner di Optuna, sono state aggiunte **dopo** il primo addestramento.

---

## 2. Modifiche al modello (versione 2)

Dopo il primo addestramento, l'analisi del codice ha evidenziato tre limiti. Le correzioni si trovano in un nuovo file **`gnn_v2.py`**: il file originale `gnn.py` non viene modificato.

| # | Limite del codice originale | Modifica | File |
|---|---|---|---|
| 1 | I dati di addestramento non vengono mai mescolati: ogni batch contiene prefissi delle stesse poche tracce, sempre nello stesso ordine | `shuffle=True` nel `DataLoader` di addestramento (validazione e test invariati) | `hyper_parameters_search.py` |
| 2 | Il readout (somma, media o massimo) mescola tutti i nodi: il modello non sa quale sia **l'ultima attività** del prefisso, l'informazione più utile per predire la successiva | Il vettore dell'ultimo nodo del prefisso viene concatenato al riassunto del grafo. Nessun'altra informazione viene aggiunta: il target non è mai visibile e ogni prefisso resta trattato da solo | `gnn_v2.py` |
| 3 | Fino a 6 layer `SAGEConv` senza normalizzazione né connessioni residue: addestramento instabile e sensibile al learning rate | `LayerNorm` dopo ogni layer `SAGEConv` e connessione residua a partire dal 2° layer | `gnn_v2.py` |

Per usare la versione 2, in `hyper_parameters_search.py`:

```python
from gnn_v2 import SAGE_GNN as GNN    # riga 12, invece di: from gnn import ...

train_loader = DataLoader(dataset=train_dataset, batch_size=parameters.get('batch_size'), shuffle=True)    # riga 146
```

Gli iperparametri esplorati da Optuna e il protocollo di addestramento restano identici a quelli della versione 1.

**Risultato**: loss di validazione migliore su tutti e 5 i dataset, F1 macro fino a **+9,5 punti** e addestramento fino a **7 volte più veloce** (dettagli nella sezione 6).

---

## 3. Esecuzione

```powershell
python -u hyper_parameters_search.py 2>&1 | Tee-Object log.txt
```

Il messaggio rosso `NativeCommandError` mostrato da PowerShell non è grave: proviene dalla barra di avanzamento.

I risultati vengono scritti in `models/`. Un dataset già completato viene saltato all'esecuzione successiva.

---

## 4. Come funziona

```
Event log  →  Instance Graphs  →  Prefix-IGs  →  GNN  →  Prossima attività
```

1. **I dati**: ogni esecuzione di un processo (una *traccia*) è rappresentata da un grafo orientato, l'**Instance Graph**. I nodi sono le attività, descritte dalla loro codifica one-hot e da 3 caratteristiche temporali; gli archi sono le loro dipendenze.
2. **I prefissi**: ogni grafo viene suddiviso in grafi parziali (le prime 2, 3, 4… attività). Ognuno è associato all'attività che segue realmente.
3. **Il modello**: da 2 a 6 layer **`SAGEConv`** (ogni attività aggrega la media dei suoi predecessori), un *readout* (somma, media o massimo) che riassume il grafo in un vettore, poi due layer lineari che predicono la prossima attività (dettagli e altre architetture nella sezione 5).
4. **L'ottimizzazione**: **Optuna** prova 20 combinazioni di iperparametri per dataset (learning rate, dimensione dei batch, numero e dimensione dei layer, dropout, readout). Ogni combinazione viene addestrata fino a 150 epoch, con early stopping dopo 20 epoch senza miglioramento. Il modello migliore viene scelto in base alla loss di validazione, poi valutato sul test set.

| File | Ruolo |
|---|---|
| `config.py` | Parametri globali |
| `data_encoding.py` | Generazione dei prefix-IGs |
| `gnn.py` | Architettura GraphSAGE (versione originale) |
| `gnn_v2.py` | Architettura GraphSAGE modificata (versione 2) |
| `gnn_GATv2Conv_v2.py` | Architettura GATv2 con le modifiche della versione 2 |
| `hyper_parameters_search.py` | Ricerca degli iperparametri, addestramento e test |

---

## 5. Architetture utilizzate

Tutti i modelli condividono lo **stesso scheletro**: cambia solo il tipo di layer che fa circolare l'informazione tra le attività. In questo modo il confronto tra architetture è equo.

### Cosa succede quando si dà un prefisso al modello

Esempio: una richiesta è in corso e sono già avvenute due attività. Quale sarà la prossima?

```
Evento A  ──►  Evento B          →  prossima attività?
```

1. **Le attività si scambiano informazioni (layer GNN, da 2 a 6 volte).** Ogni attività è descritta da una scheda di 32 numeri (quale attività è e quando è avvenuta). Ogni attività legge la scheda di quelle che la precedono e aggiorna la propria: alla fine, la scheda dell'*Evento B* dice «sono l'Evento B, e prima di me c'è stato l'Evento A».
2. **Si fa un riassunto (readout).** Le schede di tutte le attività vengono combinate in **una sola** (somma, media o massimo), che descrive l'intero prefisso. Nella versione 2 si aggiunge anche la scheda dell'**ultima attività**, la più utile.
3. **Si decide (piccola rete finale con 2 layer lineari).** La rete legge il riassunto e dà un punteggio a **ogni attività possibile**, ad esempio *Evento C* 80 %, *Evento D* 15 %, *Evento E* 5 %. L'attività con il punteggio più alto è la predizione.

Durante l'addestramento si confronta la predizione con l'attività reale: se è sbagliata, il modello corregge leggermente i suoi pesi.

### GraphSAGE — versione 1 (`gnn.py`, codice originale)

Ogni attività combina la propria scheda con la **media** delle schede delle attività che la precedono, usando due matrici di pesi distinte:

`nuovo(i) = W_sé · x(i) + W_vicini · media(predecessori di i)`

Semplice e leggero; tratta tutti i predecessori allo stesso modo.

### GraphSAGE — versione 2 (`gnn_v2.py`)

Stesso layer `SAGEConv`, con le modifiche della sezione 2: la scheda dell'**ultima attività** viene aggiunta al riassunto, una **normalizzazione** (LayerNorm) e una **connessione residua** stabilizzano i layer, e i dati di addestramento vengono **mescolati**.

### GATv2 — versione 2 (`gnn_GATv2Conv_v2.py`)

Invece di fare una media semplice, ogni attività **impara quanto pesare ciascun predecessore** (meccanismo di *attenzione*): i pesi sono calcolati dal modello e sommano a 1. Usa 4 teste di attenzione indipendenti, mediate tra loro. Stesso scheletro della versione 2 di GraphSAGE.

### Confronto

| Modello | Come combina i predecessori | Parametri* |
|---|---|---:|
| GraphSAGE v1 | Media semplice | ≈ 246 000 |
| GraphSAGE v2 | Media semplice + ultima attività | ≈ 273 000 |
| GATv2 v2 | Media pesata appresa (attenzione) | ≈ 928 000 |

\* Esempio con 3 layer nascosti da 160 neuroni, sul dataset Receipt (32 valori in ingresso, 29 attività possibili).

---

## 6. Risultati

### In sintesi

- **GraphSAGE versione 2 è il modello migliore nel complesso**: migliora la loss di validazione su **tutti e 5 i dataset** rispetto alla versione 1, con il guadagno più netto sulle attività rare (F1 macro **+9,5 punti** su Receipt), ed è molto più veloce da addestrare (BPI12WC: da **7 h 20 a 1 h 02**).
- **GATv2 non porta un guadagno chiaro** rispetto a GraphSAGE v2: leggermente migliore su BPI20P e BPI12WC, nettamente peggiore su Receipt, pari altrove, con un modello 3 volte più pesante.
- **Le modifiche della versione 2 funzionano con entrambe le architetture**: anche GATv2 v2 ha una loss di validazione migliore di GraphSAGE v1 su tutti i dataset.

### Metriche complete

Modello migliore di ogni dataset (scelto sulla loss di validazione), valutato sul test set. In **grassetto** il valore migliore per ogni dataset.

| Dataset | Modello | Val loss ↓ | Accuracy | F1 macro | F1 weighted |
|---|---|---:|---:|---:|---:|
| Receipt | GraphSAGE v1 | 0,484 | 85,87 % | 43,25 % | 83,87 % |
| | GraphSAGE v2 | **0,443** | **86,66 %** | **52,76 %** | **84,90 %** |
| | GATv2 v2 | 0,458 | 85,79 % | 45,36 % | 83,81 % |
| Helpdesk_no_resources | GraphSAGE v1 | 0,672 | 78,49 % | 29,69 % | 73,48 % |
| | GraphSAGE v2 | **0,662** | **79,16 %** | **30,50 %** | **74,52 %** |
| | GATv2 v2 | 0,669 | 78,09 % | 30,10 % | 73,49 % |
| BPI20R | GraphSAGE v1 | 0,393 | **85,46 %** | 41,66 % | 80,62 % |
| | GraphSAGE v2 | **0,389** | 85,43 % | 40,52 % | **80,76 %** |
| | GATv2 v2 | 0,392 | 85,31 % | **42,54 %** | 80,68 % |
| BPI20P | GraphSAGE v1 | 0,379 | 85,25 % | 44,36 % | 81,89 % |
| | GraphSAGE v2 | 0,363 | **85,48 %** | 46,56 % | 81,97 % |
| | GATv2 v2 | **0,361** | 85,42 % | **47,75 %** | **82,68 %** |
| BPI12WC | GraphSAGE v1 | 0,491 | 77,31 % | 58,99 % | 73,31 % |
| | GraphSAGE v2 | 0,483 | 78,06 % | 61,97 % | **75,83 %** |
| | GATv2 v2 | **0,482** | **78,17 %** | **63,45 %** | 75,71 % |

**Come leggere le metriche:**

- **Val loss**: errore sul validation set, usato per scegliere il modello (più è bassa, meglio è). Si confronta solo tra modelli dello stesso dataset.
- **Accuracy**: percentuale di predizioni corrette.
- **F1 macro**: media dell'F1 di tutte le attività, ognuna con lo stesso peso: misura le prestazioni anche sulle **attività rare**.
- **F1 weighted**: media dell'F1 pesata per la frequenza delle attività: riflette le prestazioni sulle attività più comuni.

### Osservazioni

**GraphSAGE v2 rispetto alla versione 1**

- **Loss di validazione migliore su tutti i dataset**: il confronto è valido perché la funzione di perdita è la stessa.
- **Il guadagno principale riguarda le attività rare** (F1 macro): +9,5 punti su Receipt, +3 su BPI12WC, +2,2 su BPI20P. È coerente con la modifica 2: conoscere l'ultima attività aiuta soprattutto a distinguere i casi poco frequenti.
- **Addestramento molto più rapido**: i modelli convergono prima (normalizzazione, dati mescolati) e Optuna scarta prima le combinazioni peggiori.
- **Il modello è meno sensibile agli iperparametri**: la prima combinazione provata da Optuna (identica per tutti i dataset) è la migliore su 3 dataset su 5.

**GATv2 rispetto a GraphSAGE v2**

- **Nessun guadagno chiaro**: GATv2 è leggermente migliore su BPI20P e BPI12WC (meno di 0,002 di val loss, entro il margine del caso), peggiore su Receipt (F1 macro −7,4 punti) e su Helpdesk.
- **Ipotesi da verificare**: l'attenzione serve a scegliere tra più predecessori; potrebbe essere utile solo nei processi con molte attività parallele. Il modello, 3 volte più pesante, potrebbe anche adattarsi troppo ai dati sui dataset più piccoli.

**Per tutti i modelli**

- **Sbilanciamento delle classi**: il divario tra F1 *weighted* e F1 *macro* resta ampio ovunque. Le attività frequenti sono predette bene, quelle rare molto meno.
- **Helpdesk resta il punto debole** (F1 macro ≈ 30 % per i tre modelli): nessuna delle modifiche affronta lo sbilanciamento delle classi.
- **BPI20R non cambia**: i tre modelli ottengono risultati quasi identici, il dataset sembra aver raggiunto un limite.

> **Limiti del confronto**: ogni modello è stato addestrato una sola volta, con un solo seed; scarti inferiori a circa 1 punto possono essere dovuti al caso. La versione 2 combina tre modifiche (più `n_warmup_steps=10`): l'effetto di ciascuna non è isolato.

### Prossimi addestramenti

| Addestramento | Stato |
|---|---|
| GraphSAGE versione 1 (codice originale) | ✅ completato |
| GraphSAGE versione 2 (vedi sezione 2) | ✅ completato |
| GATv2 con le modifiche della versione 2 | ✅ completato |
| Ponderazione della loss per lo sbilanciamento delle classi | in programma |
| Altre architetture GNN (GCN, GIN…) | in programma |

---

## AUTORI

- TOYEM TEZEM RYAN PARFAIT
- MICHELE MARZIONI
- GIACOMO PRINCIPI
