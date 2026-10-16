"""
Prova del modello addestrato su un prefisso: quale sarà la prossima attività?

Uso (nella cartella `code`, con il .venv attivo):
    python prova_predizione.py

Per provare un altro prefisso, un altro dataset o un altro modello,
modificare i parametri qui sotto.
"""
import json
import torch
import pandas as pd
from gnn_v2 import SAGE_GNN     # per GATv2: from gnn_GATv2Conv_v2 import GATv2_GNN as SAGE_GNN
#from gnn_GatedGraph_v3 import GGNN_GNN as SAGE_GNN


# ----- Parametri da modificare -----
DATASET = 'Helpdesk_no_resources'
PREFISSO = 's3789_3'       # <traccia>_<lunghezza del prefisso> (vedi i file in dataset/<DATASET>_tensors)
MODELLI = 'models_sage_v2'      # cartella dei risultati dell'addestramento

# ----- 1. Il prefisso da provare -----
# Nome del file: <dataset>_<traccia>_<lunghezza del prefisso>.pt
prefisso = torch.load(f'dataset/{DATASET}_tensors/{DATASET}_{PREFISSO}.pt', weights_only=False)

# ----- 2. I nomi delle attività -----
# Il file JSON associa a ogni nome il suo vettore one-hot.
# Si costruisce il dizionario inverso: posizione dell'1 -> nome dell'attività.
ohe = json.load(open(f'dataset/{DATASET}_ohe_activity.json'))
nomi = {vettore.index(1): nome for nome, vettore in ohe.items()}

# ----- 3. Il modello addestrato -----
# Il trial migliore è nella prima riga del file riassuntivo.
trial = int(pd.read_csv(f'{MODELLI}/{DATASET}_metrics.csv').iloc[0]['trial'])
salvataggio = torch.load(f'{MODELLI}/{DATASET}/model_it_{trial}.pt', weights_only=False, map_location='cpu')

modello = SAGE_GNN(data=prefisso, parameters=salvataggio['parameters'])   # modello vuoto, con la forma giusta
modello.load_state_dict(salvataggio['state_dict'])                        # si caricano i pesi appresi
modello.eval()                                                            # modalità predizione (dropout disattivato)

# ----- 4. La predizione -----
with torch.no_grad():                                  # nessun apprendimento qui
    punteggi = modello(prefisso)                       # un punteggio per ogni attività possibile
probabilita = torch.softmax(punteggi, dim=-1)[0]       # punteggi -> probabilità (somma = 100 %)

# ----- 5. Il risultato -----
attivita = [nomi[i] for i in prefisso.x[:, :len(nomi)].argmax(dim=1).tolist()]
print('Prefisso             :', ' -> '.join(attivita))
print('Attività reale       :', prefisso.next_activity, f'(insieme: {prefisso.set})')
print('Il modello propone   :')
valori, posizioni = probabilita.topk(3)                # le 3 attività più probabili
for p, i in zip(valori.tolist(), posizioni.tolist()):
    print(f'   {nomi[i]:<45} {p * 100:5.1f} %')
