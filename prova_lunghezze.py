from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score


DATASET = [
    'BPI12WC',
    'BPI20P',
    'BPI20R',
    'Helpdesk_no_resources',
    'Receipt',
]
MODELLI = {
    'GraphSAGE': 'models_sage',
    'GATv2': 'models_gatv2',
    'GGNN': 'models_ggnn',
    'GCN': 'models_gcn',
}

tabelle = []
for dataset in DATASET:
    risultati_per_modello = {}
    for nome_modello, cartella in MODELLI.items():
        file_metriche = Path(cartella) / f'{dataset}_metrics.csv'
        metriche = pd.read_csv(file_metriche)
        trial = int(metriche.loc[metriche['val'].idxmin(), 'trial'])

        file_risultati = (
            Path(cartella) / dataset / f'prefix_results_it_{trial}.csv'
        )
        risultati = pd.read_csv(file_risultati)
        colonne_richieste = {'prefix_size', 'prediction', 'label'}
        colonne_mancanti = colonne_richieste.difference(risultati.columns)
        if colonne_mancanti:
            raise ValueError(
                f'{file_risultati} non contiene le colonne richieste: '
                f'{sorted(colonne_mancanti)}'
            )

        righe = []
        for lunghezza, gruppo in risultati.groupby('prefix_size'):
            f1 = f1_score(
                gruppo['label'],
                gruppo['prediction'],
                average='macro',
                zero_division=0,
            )
            righe.append({
                'dataset': dataset,
                'modello': nome_modello,
                'lunghezza_prefisso': int(lunghezza),
                'numero_prefissi': len(gruppo),
                'f1_macro': f1 * 100,
            })

        risultati_per_modello[nome_modello] = pd.DataFrame(righe)

    tabella = pd.concat(risultati_per_modello.values(), ignore_index=True)
    tabelle.append(tabella)
    print(f'\n=== {dataset} ===')
    print(tabella.to_string(index=False, formatters={'f1_macro': '{:.2f}'.format}))

    fig, ax = plt.subplots(figsize=(10, 6))
    for nome_modello, risultati in risultati_per_modello.items():
        risultati = risultati.sort_values('lunghezza_prefisso')
        ax.plot(
            risultati['lunghezza_prefisso'],
            risultati['f1_macro'],
            marker='o',
            linewidth=2,
            label=nome_modello,
        )

    ax.set(
        title=f'F1 macro per lunghezza del prefisso - {dataset}',
        xlabel='Lunghezza del prefisso (numero di nodi)',
        ylabel='F1 macro (%)',
        ylim=(0, 100),
    )
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Modello')
    fig.tight_layout()

    file_plot = f'f1_macro_per_lunghezza_{dataset}.png'
    fig.savefig(file_plot, dpi=150)
    plt.close(fig)
    print(f'Grafico salvato in: {file_plot}')

file_csv = 'f1_macro_per_lunghezza_tutti_dataset.csv'
pd.concat(tabelle, ignore_index=True).to_csv(file_csv, index=False)
print(f'\nTabella completa salvata in: {file_csv}')
