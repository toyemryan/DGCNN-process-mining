import random
import optuna
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from os.path import join, exists
from pickle import dump, load
from os import makedirs, listdir
from data_encoding import main as data_encoding
#from gnn import SAGE_GNN as GNN
#from gnn_v2 import SAGE_GNN as GNN
#from gnn_GATv2Conv_v2 import GATv2_GNN as GNN
#from gnn_GATv2Conv_v3 import GATv2_GNN as GNN
from gnn_GatedGraph_v3 import GGNN_GNN as GNN
from config import EPOCHS, N_TRIALS, LOG_NAME, EARLY_STOPPING
from sklearn.metrics import classification_report


def compute_metrics(y_true, y_pred, set):
    def get_flat_dict(d, parent_key='', sep= '_'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(get_flat_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # filtering single class metrics
    filtered_metrics = {k: v for k, v in metrics.items() if not k[0].isdigit()}
    flatten_metrics = get_flat_dict(filtered_metrics)
    flatten_metrics ={f'{set}_{k}': round(v*100, 2) for k, v in flatten_metrics.items() if 'support' not in k}

    return flatten_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_hyperparameters(trial):
    parameters = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'batch_size': trial.suggest_int('batch_size', 128, 256, step=64),
        'hidden_layers': trial.suggest_int('hidden_layers', 0, 4),
        'layers_size': trial.suggest_int('layers_size', 64, 256, step=32),
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        'readout': trial.suggest_categorical('readout', ['add', 'mean', 'max']),
        'aggregator': trial.suggest_categorical('aggregator', ['mean']),
    }
    # iperparametri specifici di GATv2: solo quando si addestra GATv2
    if GNN.__name__ == 'GATv2_GNN':
        parameters['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
        parameters['concat'] = trial.suggest_categorical('concat', [True, False])

    # iperparametri specifici di GGNN: solo quando si addestra GGNN
    if GNN.__name__ == 'GGNN_GNN':
        parameters['ggnn_aggr'] = trial.suggest_categorical('ggnn_aggr', ['add', 'mean', 'max'])
        parameters['ggnn_input'] = trial.suggest_categorical('ggnn_input', ['pad', 'linear'])
    return parameters


def train_and_val_network(model, device, optimizer, criterion, train_loader, val_loader):
    model.train()
    train_loss = 0.0
    epoch_grad_norm = 0.0
    for batch in train_loader:
        batch = batch.to(device)

        out = model(batch)
        batch_labels = batch.y.argmax(dim=-1).squeeze()
        loss = criterion(out, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    epoch_grad_norm /= len(train_loader)
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            out = model(batch)
            batch_labels = batch.y.argmax(dim=-1).squeeze()
            loss = criterion(out, batch_labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    return train_loss, val_loss, optimizer.param_groups[0]['lr']


def test_network(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    predictions, labels = torch.tensor([], device=device), torch.tensor([], device=device)
    prefix_size = torch.tensor([], device=device)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            prefix_size = torch.cat((prefix_size, batch.prefix_size))
            out = model(batch)

            batch_predictions = torch.log_softmax(out, dim=-1).argmax(dim=-1).int()
            predictions = torch.cat((predictions, batch_predictions))
            batch_labels = batch.y.argmax(dim=-1).squeeze()

            labels = torch.cat((labels, batch_labels.int()))

            # test loss
            loss = criterion(out, batch_labels)
            test_loss += loss.item()

    # test metrics
    labels, predictions = labels.cpu().tolist(), predictions.cpu().tolist()
    test_loss /= len(test_loader)

    metrics = {'test_loss': round(test_loss, 8)}
    test_metrics = compute_metrics(labels, predictions, 'test')
    metrics.update(test_metrics)

    prefix_results = pd.DataFrame({
        'set': ['test'] * len(prefix_size),
        'prefix_size': prefix_size.cpu().tolist(),
        'prediction': predictions,
        'label': labels,
    })

    return metrics, prefix_results


def objective(trial, train_dataset, val_dataset, test_dataset, device, log_name, resumed):
    parameters = get_hyperparameters(trial)
    p_str = 'comb'
    for key, value in parameters.items():
        n_k = ''
        key_init = key.split('_')
        for item in key_init:
            n_k += item[0]

        p_str += f'_{n_k}_{value}'

    ck_path = join('various', 'optuna', f'{log_name}_ckp_{p_str}.pt')
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters.get('batch_size'), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=parameters.get('batch_size'))
    test_loader = DataLoader(dataset=test_dataset, batch_size=parameters.get('batch_size'))

    criterion = CrossEntropyLoss()

    model = GNN(data=train_dataset[0], parameters=parameters)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters.get("learning_rate"), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    epochs_done = 0
    best_val = float('inf')
    no_improvements = 0

    if resumed and exists(ck_path):
        checkpoint = torch.load(ck_path, weights_only=True)
        model.load_state_dict(checkpoint.get('model'))
        optimizer.load_state_dict(checkpoint.get('optimizer'))
        scheduler.load_state_dict(checkpoint.get('scheduler'))
        epochs_done = checkpoint.get('epochs_done')
        best_val = checkpoint.get('best_val')
        no_improvements = checkpoint.get('no_improvements')
        print(f'\nResuming combination: {trial.params}')

    else:
        print(f'\nStarting combination: {trial.params}')

    makedirs(join('models', log_name), exist_ok=True)
    metrics = {}
    for epoch in range(epochs_done, EPOCHS):
        train, val, lr = train_and_val_network(model, device, optimizer, criterion, train_loader, val_loader)

        report = f'Epoch: {epoch+1} | {lr:.3e} | Train: {train:.4f} | Val: {val:.4f}'
        if val < best_val:
            no_improvements = 0

            best_val = val
            report += ' ** Best'

            metrics, prefix_results = test_network(model, device, criterion, test_loader)
            metrics.update({'combination': p_str, 'val': val, 'epoch': epoch, 'trial': trial.number})
            df_metrics = pd.DataFrame([metrics])
            df_metrics.to_csv(join('models', log_name, f'metrics_it_{trial.number}.csv'), index=False)
            prefix_results.to_csv(join('models', log_name, f'prefix_results_it_{trial.number}.csv'), index=False)

            torch.save({
                'state_dict': model.state_dict(),
                'parameters': parameters,
                'parameters_string': p_str,
            }, join('models', log_name, f'model_it_{trial.number}.pt'))

        else:
            no_improvements += 1

        print(report)

        scheduler.step(val, epoch=epoch)
        trial.report(val, step=epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'parameters': parameters,
            'best_val': best_val,
            'epochs_done': epoch + 1,
            'no_improvements': no_improvements,
            'parameters_string': p_str,
        }, ck_path)

        if trial.should_prune() or no_improvements == EARLY_STOPPING:
            print(metrics)

            raise optuna.TrialPruned()

    print(metrics)

    return best_val


def main(log_name):
    set_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\n** Device: {device}')
    ts_path = join('dataset', f'{log_name}_tensors')

    if not exists(join(ts_path, 'done')):
        data_encoding(log_name)

    print('Reading tensors')
    prefixes = [torch.load(join(ts_path, part), weights_only=False) for part in listdir(ts_path)
               if part.startswith(f'{log_name}_') and part.endswith('.pt')]

    train_dataset = [data for data in prefixes if data.get('set') == 'train']
    val_dataset = [data for data in prefixes if data.get('set') == 'validation']
    test_dataset = [data for data in prefixes if data.get('set') == 'test']

    makedirs(join('various', 'optuna'), exist_ok=True)
    db_path, sp_path, pr_path = (join(f'various', 'optuna', f'{log_name}.db'),
                                 join(f'various', 'optuna', f'{log_name}_sampler.pkl'),
                                 join(f'various', 'optuna', f'{log_name}_pruner.pkl'))

    # load study
    if exists(db_path) and exists(sp_path) and exists(pr_path):
        with open(sp_path, 'rb') as fin:
            sampler = load(fin)

        with open(pr_path, 'rb') as fin:
            pruner = load(fin)

        study = optuna.create_study(
            study_name=f'{log_name}_optuna',
            storage=f'sqlite:///{db_path}',
            load_if_exists=True,
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )

        # queue fail trial
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.FAIL:
                study.enqueue_trial(trial.params)
        resumed = True

    # create study
    else:
        resumed = False

        study = optuna.create_study(
            study_name=f'{log_name}_optuna',
            storage=f'sqlite:///{db_path}',
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) #to 30 at 10
        )

        with open(sp_path, 'wb') as fout:
            dump(study.sampler, fout)

        with open(pr_path, 'wb') as fout:
            dump(study.pruner, fout)

    completed = len([trial for trial in study.trials
                 if trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, test_dataset, device, log_name, resumed),
                   n_trials=N_TRIALS - completed, show_progress_bar=True, gc_after_trial=True, catch=(Exception,))

    all_results = pd.DataFrame([])
    results_csv = [join('models', log_name, result_csv) for result_csv in
                   listdir(join('models', log_name)) if result_csv.startswith('metrics_') and result_csv.endswith('.csv')]

    for result_csv in results_csv:
        all_results = pd.concat([all_results, pd.read_csv(result_csv)])

    all_results.sort_values(by=['val'], ascending=True, inplace=True)
    all_results.to_csv(join('models', f'{log_name}_metrics.csv'), index=False)


if __name__ == '__main__':
    for dataset in reversed(sorted(LOG_NAME)):
        try:
            print(f'** Dataset: {dataset}')
            if exists(join('models', f'{dataset}_metrics.csv')):
                print('Completed')
                continue

            main(dataset)

        except Exception:
            import traceback
            traceback.print_exc()
