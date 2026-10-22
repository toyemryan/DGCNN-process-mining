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

## 2. Esecuzione

```powershell
python -u hyper_parameters_search.py 2>&1 | Tee-Object log.txt
```

Il messaggio rosso `NativeCommandError` mostrato da PowerShell non è grave: proviene dalla barra di avanzamento.

I risultati vengono scritti in `models/`. Un dataset già completato viene saltato all'esecuzione successiva.

---

## 3. Come funziona

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
| `gnn_v2.py` | Architettura GraphSAGE |
| `gnn_GATv2Conv_v3.py` | GATv2: teste di attenzione scelte da Optuna |
| `gnn_GatedGraph_v3.py` | GGNN: GRU, aggregazione e ingresso scelti da Optuna |
| `gnn_GCNConv.py` | GCN (scheletro originale) |
| `hyper_parameters_search.py` | Ricerca degli iperparametri, addestramento e test |
| `prova_predizione.py` | Prova del modello addestrato su un prefisso (sezione 6) |
| `migliori_iperparametri.py` | Metriche e iperparametri scelti da Optuna per ogni modello (sezione 7) |

---

## 4. Architetture utilizzate

### Cosa succede quando si dà un prefisso al modello

Esempio: una richiesta è in corso e sono già avvenute due attività. Quale sarà la prossima?

```
Evento A  ──►  Evento B          →  prossima attività?
```

1. **Le attività si scambiano informazioni (layer GNN, da 2 a 6 volte).** Ogni attività è descritta da una scheda di 32 numeri (quale attività è e quando è avvenuta). Ogni attività legge la scheda di quelle che la precedono e aggiorna la propria: alla fine, la scheda dell'*Evento B* dice «sono l'Evento B, e prima di me c'è stato l'Evento A».
2. **Si fa un riassunto (readout).** Le schede di tutte le attività vengono combinate in **una sola** (somma, media o massimo), che descrive l'intero prefisso.
3. **Si decide (piccola rete finale con 2 layer lineari).** La rete legge il riassunto e dà un punteggio a **ogni attività possibile**, ad esempio *Evento C* 80 %, *Evento D* 15 %, *Evento E* 5 %. L'attività con il punteggio più alto è la predizione.

Durante l'addestramento si confronta la predizione con l'attività reale: se è sbagliata, il modello corregge leggermente i suoi pesi.

### GraphSAGE (`gnn.py`)

La scheda dell'**ultima attività** viene aggiunta al riassunto, una **normalizzazione** (LayerNorm) e una **connessione residua** stabilizzano i layer, e i dati di addestramento vengono **mescolati**.

### GATv2 (`gnn_GATv2Conv.py`): le teste scelte da Optuna

Ogni **testa** di attenzione calcola il proprio modo di pesare i predecessori. Le teste si possono combinare in due modi:

| `concat` | Combinazione | Dimensione di ogni testa | Parametri* |
|---|---|---|---:|
| `False` | **media** delle teste | `layers_size` | ≈ 276 000 × numero di teste |
| `True` | teste **messe in fila** | `layers_size ÷ num_heads` | ≈ 276 000, qualunque sia il numero di teste |

**NB :** è stato aggiunto la **normalizzazione** (LayerNorm) e la **connessione residua** dall'architettura.

### GGNN (`gnn_GatedGraph.py`)

Ad ogni passo di propagazione, ogni attività riceve i messaggi dei suoi predecessori e aggiorna la propria scheda con una **GRU**, che decide da sola quanto conservare della scheda precedente e quanto prendere della nuova informazione. È il layer usato da SR-GNN (Wu et al., AAAI 2019) per predire il prossimo elemento di una sessione, un problema molto simile al nostro.

Un solo layer `GatedGraphConv` con `hidden_layers + 2` passi (stesso numero degli altri modelli). La GRU sostituisce la connessione residua. Due scelte lasciate a Optuna:

| Iperparametro | Opzioni | Significato |
|---|---|---|
| `ggnn_aggr` | `add`, `mean`, `max` | Come si combinano i messaggi dei predecessori |
| `ggnn_input` | `pad`, `linear` | Le 32 schede in ingresso completate con zeri, oppure trasformate da un layer appreso |

**NB :** è stato aggiunto la **normalizzazione** (LayerNorm) e la **connessione residua** dall'architettura.

### GCN (`gnn_GCNConv.py`)

Ogni attività fa una **media pesata** della propria scheda e di quelle dei predecessori, con **una sola matrice di pesi** per tutti: non distingue «chi sono» da «cosa è successo prima di me». La normalizzazione dipende dal numero di collegamenti di ogni attività ed è pensata per grafi **non orientati**. Usa lo scheletro originale (senza ultima attività, LayerNorm e residuo).

Optuna sceglie anche `improved` (l'attività conta il doppio di sé stessa), ma in PyG 2.8 questo parametro **non ha effetto** se non si passa `edge_weight`: le uscite sono identiche con `True` e `False`.

### Confronto

| Modello | Come combina i predecessori | Parametri* |
|---|---|---:|
| GraphSAGE | Media semplice + ultima attività | ≈ 273 000 |
| GATv2 | Attenzione, teste scelte da Optuna | ≈ 276 000 – 928 000 |
| GGNN | Somma / media / massimo + GRU | ≈ 339 000 – 344 000 |
| GCN | Media pesata, una sola matrice (scheletro originale) | ≈ 139 000 |

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
if GNN.__name__ == 'GCN_GNN':
    parameters['improved'] = trial.suggest_categorical('improved', [True, False])
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

Per GATv2, sostituire l'import in cima allo script con `from gnn_GATv2Conv import GATv2_GNN as SAGE_GNN` e usare `MODELLI = 'models_gat_v2'`.

> Lo script indica l'insieme del prefisso (`train`, `validation` o `test`). Per una prova onesta, scegliere un prefisso del **test set**, che il modello non ha mai visto durante l'addestramento.

---

## 7. Risultati

### In sintesi

- **GraphSAGE, GATv2 e GGNN sono molto vicini tra loro**: a parità di scheletro, la scelta del layer cambia poco i risultati.
- **GCN è ultimo in val loss su tutti i dataset**, con gli scarti maggiori su Receipt e BPI20P.
- **GGNN è il più regolare** (rango medio 1,6 su 4, mai oltre il 2° posto) e ha la migliore accuracy su 3 dataset. **GraphSAGE** vince su Receipt e Helpdesk.
- **Sui dataset BPI gli scarti tra GraphSAGE, GATv2 e GGNN sono minimi** (≤ 0,003 di val loss): nessun vincitore netto.
- **Helpdesk resta bloccato** a ≈ 30 % di F1 macro con tutti i modelli: è un limite dei dati (prefissi identici portano ad attività diverse), non dell'architettura.

### Metriche

Modello migliore di ogni dataset (scelto sulla loss di validazione), valutato sul test set. In **grassetto** il migliore di ogni riga.

**Val loss** (più bassa = meglio)

| Dataset | SAGE | GATv2 | GGNN | GCN |
| --- | ---: | ---: | ---: | ---: |
| Receipt | **0,4433** | 0,4539 | 0,4480 | 0,5509 |
| Helpdesk_no_resources | **0,6624** | 0,6708 | 0,6628 | 0,6803 |
| BPI20R | 0,3894 | 0,3908 | **0,3884** | 0,4030 |
| BPI20P | 0,3633 | 0,3653 | **0,3624** | 0,4151 |
| BPI12WC | 0,4833 | **0,4826** | 0,4828 | 0,4943 |

**Accuracy (%)**

| Dataset | SAGE | GATv2 | GGNN | GCN |
| --- | ---: | ---: | ---: | ---: |
| Receipt | 86,66 | 86,24 | **86,70** | 84,33 |
| Helpdesk_no_resources | **79,16** | 78,97 | 78,67 | 78,09 |
| BPI20R | 85,43 | 85,45 | **85,51** | 85,09 |
| BPI20P | 85,48 | 85,44 | **85,87** | 83,92 |
| BPI12WC | 78,06 | **78,33** | 78,20 | 77,51 |

**F1 macro (%)** — tutte le attività con lo stesso peso, comprese le rare

| Dataset | SAGE | GATv2 | GGNN | GCN |
| --- | ---: | ---: | ---: | ---: |
| Receipt | **52,76** | 45,15 | 51,48 | 41,99 |
| Helpdesk_no_resources | **30,50** | 30,36 | 30,16 | 29,56 |
| BPI20R | 40,52 | 42,59 | **42,60** | 42,05 |
| BPI20P | 46,56 | 47,35 | **48,14** | 45,57 |
| BPI12WC | 61,97 | **64,05** | 63,31 | 60,35 |

**F1 weighted (%)** — attività pesate per frequenza

| Dataset | SAGE | GATv2 | GGNN | GCN |
| --- | ---: | ---: | ---: | ---: |
| Receipt | 84,90 | 84,22 | **85,09** | 82,25 |
| Helpdesk_no_resources | **74,52** | 74,25 | 73,91 | 73,11 |
| BPI20R | 80,76 | 80,71 | **80,85** | 80,43 |
| BPI20P | 81,97 | 82,63 | **83,03** | 80,84 |
| BPI12WC | 75,83 | **76,35** | 75,91 | 74,23 |

### Interpretazione

- **GraphSAGE**: migliore in val loss su Receipt e Helpdesk, e migliore in F1 macro su entrambi (52,8 % su Receipt): tratta meglio le attività rare sui dataset più piccoli.
- **GGNN**: la GRU dà il modello più costante, il migliore in val loss su BPI20R e BPI20P e in accuracy su Receipt, BPI20R e BPI20P.
- **GATv2**: l'attenzione non porta un guadagno chiaro. Su Receipt ogni attività ha un solo predecessore, quindi l'attenzione sceglie solo tra l'attività stessa e il suo predecessore. GATv2 è invece il migliore su BPI12WC in tutte e quattro le metriche.
- **GCN**: ultimo su tutti i dataset in val loss, con gli scarti maggiori su Receipt (+0,108) e BPI20P (+0,053), dove gli altri modelli sfruttano meglio i dati; su Helpdesk lo scarto è piccolo, perché tutti i modelli sono già al limite dei dati. Poiché cambia anche lo scheletro, lo scarto non si può attribuire al solo layer `GCNConv`.

### Perché queste differenze

*Ipotesi coerenti con i risultati.*

- **GraphSAGE, GATv2 e GGNN sono vicini** perché gli Instance Graph sono quasi delle catene (in Receipt ogni attività ha un solo predecessore). Con un solo predecessore, media (SAGE), attenzione (GATv2) e somma (GGNN) fanno quasi la stessa cosa: combinare l'attività con quella precedente. Inoltre i tre modelli condividono lo stesso scheletro e sono tutti vicini al limite dei dati.
- **GGNN è il più regolare** grazie alla GRU: invece di sommare sempre per intero vecchio vettore e correzione (come il residuo di SAGE), decide valore per valore quanto conservare e quanto aggiornare, e può anche dimenticare. Applicata ad ogni passo, legge la traccia attività dopo attività come una rete ricorrente, il modello classico per la predizione della prossima attività.
- **GCN è il meno buono** per tre cause che si sommano:
  1. una sola matrice di pesi: il nodo e il suo predecessore vengono mescolati prima di essere trasformati, e su una catena orientata il predecessore pesa più del nodo stesso (0,71 contro 0,50);
  2. nessuna connessione residua né LayerNorm: l'identità del nodo si perde layer dopo layer (Optuna sceglie infatti 0 layer nascosti su Receipt e BPI12WC);
  3. nessun ultimo nodo: il modello non vede direttamente l'attività in corso.

  Lo scarto è massimo su Receipt e BPI20P, che hanno molte attività rare (F1 macro su Receipt: 42,0 % contro 52,8 % di SAGE), e minimo su Helpdesk, dove tutti i modelli sono già al limite dei dati.

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

| Dataset | GATv2: teste / concat | GGNN: aggregazione / ingresso |
| --- | --- | --- |
| Receipt | 1 / False | mean / pad |
| Helpdesk_no_resources | 2 / True | add / linear |
| BPI20R | 2 / True | max / pad |
| BPI20P | 2 / False | mean / linear |
| BPI12WC | 1 / False | mean / linear |

**Tendenze comuni** (15 modelli migliori: GraphSAGE, GATv2 e GGNN × 5 dataset)

| Iperparametro | Scelta prevalente | Cosa significa |
| --- | --- | --- |
| `num_heads` (GATv2) | 1 o 2, **mai 4** | Bastano poche teste: i nodi hanno pochi predecessori tra cui scegliere |
| `readout` | `max` (8/15) o `add` (6/15), `mean` solo 1/15 | Conta l'attività più marcata o il numero di attività, non la loro media; BPI20P (3/3) e BPI20R (2/3) preferiscono `add` |
| `batch_size` | 256 (11/15) | I batch più grandi sono preferiti |
| `learning_rate` | tra 1e-4 e 1e-3 (13/15) | Zona di apprendimento stabile |
| `hidden_layers` | GraphSAGE: sempre 3 o 4; GATv2 e GGNN: 0 o 1 in 7 casi su 10 | GraphSAGE sfrutta reti più profonde; GATv2 e GGNN preferiscono reti poco profonde |
| `layers_size` | 160 – 224 (11/15) | Dimensione media dello spazio esplorato |

**GCN**: `readout = max` su tutti e 5 i dataset, dropout molto basso (≤ 0,08), learning rate spesso più alto (fino a 3,6e-3), 0 layer nascosti su Receipt e BPI12WC e 3 su BPI20R e BPI20P. Le scelte di `improved` (3 `False`, 2 `True`) sono casuali, perché il parametro non ha effetto.

**Cosa significa**: le stesse combinazioni ritornano spesso tra modelli diversi. Con `seed=0`, i primi trial di Optuna sono estratti a caso ma sempre allo stesso modo: il trial 0 (`lr 4,4e-4`, batch 256, 3 layer da 160, dropout 0,17, `max`) è il migliore 5 volte su 15. Con 20 trial la ricerca esplora solo una parte dello spazio, e gli scarti tra combinazioni sono piccoli: non esiste un ottimo netto, ma una **zona di buoni iperparametri**. I modelli sono poco sensibili alla scelta precisa al suo interno.

> **Limiti del confronto**: ogni modello è stato addestrato una sola volta, con un solo seed; scarti inferiori a circa 1 punto (o 0,005 di val loss) possono essere dovuti al caso. GATv2 e GGNN hanno 2 iperparametri in più di GraphSAGE. GCN usa lo scheletro originale: lo confrontiamo come modello completo, non come singolo layer.

---

## AUTORI

- TOYEM TEZEM RYAN PARFAIT
- MICHELE MARZIONI
- GIACOMO PRINCIPI
  