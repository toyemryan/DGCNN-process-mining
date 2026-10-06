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
Event log  →  Instance Graphs  →  Prefix-IGs  →  GraphSAGE  →  Prossima attività
```

1. **I dati**: ogni esecuzione di un processo (una *traccia*) è rappresentata da un grafo orientato, l'**Instance Graph**. I nodi sono le attività, descritte dalla loro codifica one-hot e da 3 caratteristiche temporali; gli archi sono le loro dipendenze.
2. **I prefissi**: ogni grafo viene suddiviso in grafi parziali (le prime 2, 3, 4… attività). Ognuno è associato all'attività che segue realmente.
3. **Il modello**: da 2 a 6 layer **`SAGEConv`** (ogni attività aggrega la media dei suoi predecessori), un *readout* (somma, media o massimo) che riassume il grafo in un vettore, poi due layer lineari che predicono la prossima attività.
4. **L'ottimizzazione**: **Optuna** prova 20 combinazioni di iperparametri per dataset (learning rate, dimensione dei batch, numero e dimensione dei layer, dropout, readout). Ogni combinazione viene addestrata fino a 150 epoch, con early stopping dopo 20 epoch senza miglioramento. Il modello migliore viene scelto in base alla loss di validazione, poi valutato sul test set.

| File | Ruolo |
|---|---|
| `config.py` | Parametri globali |
| `data_encoding.py` | Generazione dei prefix-IGs |
| `gnn.py` | Architettura GraphSAGE |
| `hyper_parameters_search.py` | Ricerca degli iperparametri, addestramento e test |

---

## 4. Risultati

### Addestramento 1 — GraphSAGE (codice originale)

Modello migliore di ogni dataset, valutato sul test set:

| Dataset | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|
| Receipt | 85,87 % | 43,25 % | 83,87 % |
| Helpdesk_no_resources | 78,49 % | 29,69 % | 73,48 % |
| BPI20R | 85,46 % | 41,66 % | 80,62 % |
| BPI20P | 85,25 % | 44,36 % | 81,89 % |
| BPI12WC | 77,31 % | 58,99 % | 73,31 % |

La prossima attività viene predetta correttamente nel 77-86 % dei casi. Il divario tra F1 *weighted* e F1 *macro* indica uno **sbilanciamento delle classi**: le attività frequenti sono predette bene, quelle rare molto meno.

Durata: una notte di addestramento su RTX 4070 Laptop.

### Prossimi addestramenti

| Addestramento | Stato |
|---|---|
| GraphSAGE (codice modificato) | in arrivo |
| Altre architetture GNN (GCN, GAT, GIN…) | in arrivo |

---

## AUTORI

- TOYEM TEZEM RYAN PARFAIT
- MICHELE MARZIONI
- GIACOMO PRINCIPI
