"""
Mostra gli iperparametri scelti da Optuna per il modello migliore di ogni dataset e di ogni esperimento.

Uso (nella cartella `code`):
    python migliori_iperparametri.py

Gli iperparametri vengono letti dal file del modello salvato (model_it_<trial>.pt),
che contiene il dizionario `parameters` usato per costruirlo.
"""
from os.path import exists
import pandas as pd
import torch

# Cartelle dei risultati da confrontare (quelle che non esistono vengono saltate)
ESPERIMENTI = ['models_sage', 'models_gatv2', 'models_ggnn', 'models_gcn']
DATASET = ['Receipt', 'Helpdesk_no_resources', 'BPI20R', 'BPI20P', 'BPI12WC']

righe = []
for esperimento in ESPERIMENTI:
    for dataset in DATASET:
        riassunto = f'{esperimento}/{dataset}_metrics.csv'
        if not exists(riassunto):
            continue

        migliore = pd.read_csv(riassunto).iloc[0]            # prima riga = trial migliore
        trial = int(migliore['trial'])
        modello = torch.load(f'{esperimento}/{dataset}/model_it_{trial}.pt', weights_only=False, map_location='cpu')

        riga = {'esperimento': esperimento.replace('models_', ''), 'dataset': dataset, 'trial': trial}
        riga.update(modello['parameters'])                   # gli iperparametri scelti da Optuna
        riga['val'] = round(migliore['val'], 4)
        riga['accuracy'] = migliore['test_accuracy']
        riga['f1_macro'] = migliore['test_macro avg_f1-score']
        riga['f1_weighted'] = migliore['test_weighted avg_f1-score']
        righe.append(riga)

tabella = pd.DataFrame(righe)
if 'learning_rate' in tabella:
    tabella['learning_rate'] = tabella['learning_rate'].map('{:.1e}'.format)
if 'dropout' in tabella:
    tabella['dropout'] = tabella['dropout'].round(2)
if 'num_heads' in tabella:
    tabella['num_heads'] = tabella['num_heads'].map(lambda v: '-' if pd.isna(v) else int(v))

pd.set_option('display.width', 250)
metriche = ['esperimento', 'dataset', 'trial', 'val', 'accuracy', 'f1_macro', 'f1_weighted']
iperparametri = ['esperimento', 'dataset'] + [c for c in tabella.columns if c not in metriche]

print('=== METRICHE (test set) ===')
print(tabella[metriche].sort_values(['dataset', 'val']).to_string(index=False))
print('\n=== IPERPARAMETRI SCELTI DA OPTUNA ===')
print(tabella[iperparametri].sort_values(['dataset', 'esperimento']).fillna('-').to_string(index=False))

tabella.to_csv('migliori_iperparametri.csv', index=False)
print('\nTabella completa salvata in migliori_iperparametri.csv')
