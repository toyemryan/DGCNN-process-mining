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
| `gnn_GATv2Conv_v3.py` | GATv2 v3: teste di attenzione scelte da Optuna |
| `gnn_GatedGraph_v3.py` | GGNN v3: GRU, aggregazione e ingresso scelti da Optuna |
| `hyper_parameters_search.py` | Ricerca degli iperparametri, addestramento e test |
| `prova_predizione.py` | Prova del modello addestrato su un prefisso (sezione 6) |
| `migliori_iperparametri.py` | Metriche e iperparametri scelti da Optuna per ogni modello (sezione 7) |

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

Invece di fare una media semplice, ogni attività **impara quanto pesare ciascun predecessore** (meccanismo di *attenzione*): i pesi sono calcolati dal modello e sommano a 1. Versione 2: 4 teste di attenzione fisse, mediate tra loro. Rispetto al file originale è stato corretto un errore: `aggr='mean'` divideva una seconda volta i pesi di attenzione, che quindi non sommavano più a 1.

### GATv2 — versione 3 (`gnn_GATv2Conv_v3.py`): le teste scelte da Optuna

Ogni **testa** di attenzione calcola il proprio modo di pesare i predecessori. Le teste si possono combinare in due modi:

| `concat` | Combinazione | Dimensione di ogni testa | Parametri* |
|---|---|---|---:|
| `False` | **media** delle teste | `layers_size` | ≈ 276 000 × numero di teste |
| `True` | teste **messe in fila** | `layers_size ÷ num_heads` | ≈ 276 000, qualunque sia il numero di teste |

**Perché farle variare:** la versione 2 (4 teste mediate, ≈ 928 000 parametri, 3,4 volte GraphSAGE v2) era peggiore sui dataset piccoli (Receipt, Helpdesk). Ipotesi: modello troppo pesante. Si è quindi lasciato a Optuna la scelta di `num_heads` ∈ {1, 2, 4} e `concat` ∈ {True, False}. Il valore 3 è escluso perché con `concat=True` non divide le dimensioni esplorate (es. 160 ÷ 3 non è intero).

### GGNN — versione 3 (`gnn_GatedGraph_v3.py`)

Ad ogni passo di propagazione, ogni attività riceve i messaggi dei suoi predecessori e aggiorna la propria scheda con una **GRU**, che decide da sola quanto conservare della scheda precedente e quanto prendere della nuova informazione. È il layer usato da SR-GNN (Wu et al., AAAI 2019) per predire il prossimo elemento di una sessione, un problema molto simile al nostro.

Un solo layer `GatedGraphConv` con `hidden_layers + 2` passi (stesso numero degli altri modelli). La GRU sostituisce la connessione residua. Due scelte lasciate a Optuna:

| Iperparametro | Opzioni | Significato |
|---|---|---|
| `ggnn_aggr` | `add`, `mean`, `max` | Come si combinano i messaggi dei predecessori |
| `ggnn_input` | `pad`, `linear` | Le 32 schede in ingresso completate con zeri, oppure trasformate da un layer appreso |

### Confronto

| Modello | Come combina i predecessori | Parametri* |
|---|---|---:|
| GraphSAGE v2 | Media semplice + ultima attività | ≈ 273 000 |
| GATv2 v2 | Attenzione, 4 teste mediate | ≈ 928 000 |
| GATv2 v3 | Attenzione, teste scelte da Optuna | ≈ 276 000 – 928 000 |
| GGNN v3 | Somma / media / massimo + GRU | ≈ 339 000 – 344 000 |

\* Esempio con 3 layer nascosti da 160 neuroni, sul dataset Receipt (32 valori in ingresso, 29 attività possibili).

### Iperparametri cercati da Optuna

Comuni a tutti i modelli: `learning_rate` (1e-5 – 1e-2), `batch_size` (128, 192, 256), `hidden_layers` (0 – 4), `layers_size` (64 – 256), `dropout` (0 – 0,4), `readout` (somma, media, massimo). Quelli specifici vengono aggiunti solo per il modello interessato, in `get_hyperparameters`:

```python
if GNN.__name__ == 'GATv2_GNN':
    parameters['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
    parameters['concat'] = trial.suggest_categorical('concat', [True, False])
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

Esempio con GraphSAGE v2 su Receipt (prefisso di 3 nodi della traccia `case-10810`):

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
| `MODELLI` | `'models_sage_v2'` | La cartella dei risultati dell'addestramento |

Per elencare i prefissi disponibili di una traccia:

```powershell
Get-ChildItem dataset\Receipt_tensors -Filter "Receipt_case-10810_*" -Name
```

Per GATv2, sostituire l'import in cima allo script con `from gnn_GATv2Conv_v2 import GATv2_GNN as SAGE_GNN` e usare `MODELLI = 'models_gat_v2'`.

> Lo script indica l'insieme del prefisso (`train`, `validation` o `test`). Per una prova onesta, scegliere un prefisso del **test set**, che il modello non ha mai visto durante l'addestramento.

---

## 7. Risultati

### In sintesi

- **I quattro modelli sono molto vicini tra loro**: a parità di scheletro (versione 2), la scelta dell'architettura cambia poco i risultati.
- **GGNN v3 è il più regolare** (rango medio 2,0 su 4, mai ultimo) e ha la migliore accuracy su 3 dataset. **GraphSAGE v2** vince sui due dataset più piccoli ed è il più semplice.
- **Sui dataset BPI gli scarti sono minimi** (≤ 0,004 di val loss): nessun vincitore netto.
- **Helpdesk resta bloccato** a ≈ 30 % di F1 macro con tutti i modelli: è un limite dei dati (sbilanciamento delle classi), non dell'architettura.

### Metriche

Modello migliore di ogni dataset (scelto sulla loss di validazione), valutato sul test set. In **grassetto** il migliore di ogni riga.

**Val loss** (più bassa = meglio)

| Dataset | SAGE v2 | GATv2 v2 | GATv2 v3 | GGNN v3 |
|---|---:|---:|---:|---:|
| Receipt | **0,4433** | 0,4576 | 0,4539 | 0,4480 |
| Helpdesk_no_resources | **0,6624** | 0,6686 | 0,6708 | 0,6628 |
| BPI20R | 0,3894 | 0,3924 | 0,3908 | **0,3884** |
| BPI20P | 0,3633 | **0,3614** | 0,3653 | 0,3624 |
| BPI12WC | 0,4833 | **0,4823** | 0,4826 | 0,4828 |

**Accuracy (%)**

| Dataset | SAGE v2 | GATv2 v2 | GATv2 v3 | GGNN v3 |
|---|---:|---:|---:|---:|
| Receipt | 86,66 | 85,79 | 86,24 | **86,70** |
| Helpdesk_no_resources | **79,16** | 78,09 | 78,97 | 78,67 |
| BPI20R | 85,43 | 85,31 | 85,45 | **85,51** |
| BPI20P | 85,48 | 85,42 | 85,44 | **85,87** |
| BPI12WC | 78,06 | 78,17 | **78,33** | 78,20 |

**F1 macro (%)** — tutte le attività con lo stesso peso, comprese le rare

| Dataset | SAGE v2 | GATv2 v2 | GATv2 v3 | GGNN v3 |
|---|---:|---:|---:|---:|
| Receipt | **52,76** | 45,36 | 45,15 | 51,48 |
| Helpdesk_no_resources | **30,50** | 30,10 | 30,36 | 30,16 |
| BPI20R | 40,52 | 42,54 | 42,59 | **42,60** |
| BPI20P | 46,56 | 47,75 | 47,35 | **48,14** |
| BPI12WC | 61,97 | 63,45 | **64,05** | 63,31 |

**F1 weighted (%)** — attività pesate per frequenza

| Dataset | SAGE v2 | GATv2 v2 | GATv2 v3 | GGNN v3 |
|---|---:|---:|---:|---:|
| Receipt | 84,90 | 83,81 | 84,22 | **85,09** |
| Helpdesk_no_resources | **74,52** | 73,49 | 74,25 | 73,91 |
| BPI20R | 80,76 | 80,68 | 80,71 | **80,85** |
| BPI20P | 81,97 | 82,68 | 82,63 | **83,03** |
| BPI12WC | 75,83 | 75,71 | **76,35** | 75,91 |

### Interpretazione

- **GraphSAGE v2**: il modello più semplice, migliore su Receipt e Helpdesk, e il migliore in F1 macro su Receipt (52,8 %).
- **GGNN v3**: la GRU dà il modello più costante, il migliore in accuracy su Receipt, BPI20R e BPI20P.
- **GATv2**: l'attenzione non porta un guadagno chiaro. Su Receipt nessuna attività ha più di un predecessore, quindi l'attenzione non ha nulla da scegliere. Su BPI20P e BPI12WC, che hanno attività parallele, GATv2 v2 è primo in val loss.
- **Teste di attenzione (v3 rispetto a v2)**: meglio su Receipt e BPI20R, peggio su Helpdesk e BPI20P, uguale su BPI12WC. Il numero di teste non è decisivo.

### Limite dei dati

**Un paradosso: stesso passato, futuro diverso.** Due prefissi di Helpdesk con la stessa sequenza di attività portano, nella realtà, ad attività diverse:

| Prefisso | Sequenza | Attività reale | GGNN v3: s6 / s9 |
|---|---|---|---|
| `s1937_3` (validation) | `artificial_start → s1 → s8` | **s6** | 75,9 % / 20,6 % |
| `s3789_3` (test) | `artificial_start → s1 → s8` | **s9** | 76,7 % / 19,0 % |

Il modello dà quasi le stesse probabilità ai due prefissi (anche GraphSAGE v2: 72,7 % / 23,0 %) e predice `s6` in entrambi i casi: giusto per il primo, sbagliato per il secondo. Nei dati, delle **3 055 tracce** che iniziano con `s1 → s8`, il **76,8 %** prosegue con `s6` e il **20,1 %** con `s9`. Due architetture diverse ritrovano esattamente queste frequenze: sono **ben calibrate**, ma nessuna può superare il 76,8 % su questo prefisso, perché nulla nei dati distingue i due casi.

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

| Dataset | GATv2 v3: teste / concat | GGNN v3: aggregazione / ingresso |
|---|---|---|
| Receipt | 1 / False | mean / pad |
| Helpdesk_no_resources | 2 / True | add / linear |
| BPI20R | 2 / True | max / pad |
| BPI20P | 2 / False | mean / linear |
| BPI12WC | 1 / False | mean / linear |

**Tendenze comuni** (20 modelli migliori: 4 modelli × 5 dataset)

| Iperparametro | Scelta prevalente | Cosa significa |
|---|---|---|
| `num_heads` (GATv2 v3) | 1 o 2, **mai 4** | La configurazione pesante della v2 non viene mai scelta: bastano poche teste |
| `readout` | `max` (10/20) o `add` (7/20), `mean` solo 3/20 | Conta l'attività più marcata o il numero di attività, non la loro media; BPI20P e BPI20R preferiscono `add` |
| `batch_size` | 256 (13/20) | I batch più grandi sono preferiti |
| `learning_rate` | tra 1e-4 e 1e-3 (16/20) | Zona di apprendimento stabile |
| `hidden_layers` | 0 o 1 nella metà dei casi | Reti poco profonde bastano: i prefissi sono corti |
| `layers_size` | 160 – 224 (15/20) | Dimensione media dello spazio esplorato |

**Cosa significa**: le stesse combinazioni ritornano spesso tra modelli diversi. Con `seed=0`, i primi trial di Optuna sono estratti a caso ma sempre allo stesso modo: il trial 0 (`lr 4,4e-4`, batch 256, 3 layer da 160, dropout 0,17, `max`) è il migliore 6 volte. Con 20 trial la ricerca esplora solo una parte dello spazio, e gli scarti tra combinazioni sono piccoli: non esiste un ottimo netto, ma una **zona di buoni iperparametri**. I modelli sono poco sensibili alla scelta precisa al suo interno.

> **Limiti del confronto**: ogni modello è stato addestrato una sola volta, con un solo seed; scarti inferiori a circa 1 punto (o 0,005 di val loss) possono essere dovuti al caso. GATv2 v3 e GGNN v3 hanno 2 iperparametri in più degli altri modelli.

---

## AUTORI

- TOYEM TEZEM RYAN PARFAIT
- MICHELE MARZIONI
- GIACOMO PRINCIPI
