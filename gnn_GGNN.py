"""
GGNN versione 3: come gnn_GatedGraph_v2.py, ma due scelte specifiche di GGNN
vengono lasciate a Optuna (iperparametri 'ggnn_aggr' e 'ggnn_input').

  ggnn_aggr  : come si combinano i messaggi dei predecessori ad ogni passo
               'add'  = somma (GGNN originale), 'mean' = media, 'max' = massimo
  ggnn_input : come le schede in ingresso (es. 32 valori) arrivano a layers_size valori
               'pad'    = completate con zeri (GGNN originale)
               'linear' = trasformate da un layer lineare appreso prima della propagazione

Invariato rispetto alla versione 2: (hidden_layers + 2) passi di propagazione con GRU,
[MODIFICA 2] ultimo nodo aggiunto al riassunto, [MODIFICA 4] LayerNorm, readout, dropout,
i due layer lineari finali. Il parametro 'aggregator' non viene usato da questo modello.
"""
import torch
from torch.nn import Module, LayerNorm
from torch.nn.functional import relu, dropout
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter


class GGNN_GNN(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_steps = parameters.get('hidden_layers') + 2      # stesso numero di passi degli altri modelli
        self.ggnn_aggr = parameters.get('ggnn_aggr', 'add')       # scelto da Optuna ('add' se assente)
        self.ggnn_input = parameters.get('ggnn_input', 'pad')     # scelto da Optuna ('pad' se assente)

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        # Ingresso: proiezione lineare appresa, oppure completamento con zeri (fatto da GatedGraphConv)
        if self.ggnn_input == 'linear':
            self.input_proj = Linear(self.input_size, self.layers_size)
        else:
            self.input_proj = None
            if self.input_size > self.layers_size:
                raise ValueError('Con ggnn_input="pad" serve layers_size >= numero di valori in ingresso')

        # GGNN: num_steps passi di propagazione, ognuno seguito dall'aggiornamento GRU
        self.ggnn = GatedGraphConv(out_channels=self.layers_size, num_layers=self.num_steps,
                                   aggr=self.ggnn_aggr)

        # [MODIFICA 4] normalizzazione dopo la propagazione
        self.norm = LayerNorm(self.layers_size)

        # [MODIFICA 2] lin1 riceve il riassunto del grafo + il vettore dell'ultimo nodo (2 x layers_size)
        self.lin1 = Linear(2 * self.layers_size, self.layers_size)
        self.lin2 = Linear(self.layers_size, self.output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:  # un solo grafo (fuori dal DataLoader)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.input_proj is not None:
            x = self.input_proj(x)                                        # ingresso 'linear'

        x = self.ggnn(x, edge_index)                                      # propagazione + GRU
        x = self.norm(x)                                                  # [MODIFICA 4]
        x = dropout(x, p=self.dropout, training=self.training)

        graph = self.readout(x, batch)                                    # riassunto del grafo

        # [MODIFICA 2] ultimo nodo di ogni grafo del batch (indice più alto = attività più recente)
        node_idx = torch.arange(x.size(0), device=x.device)
        last_idx = scatter(node_idx, batch, dim=0, reduce='max')
        last = x[last_idx]

        x = torch.cat([graph, last], dim=-1)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        return out
