"""
GATv2 con le stesse modifiche di gnn_v2.py (versione 2).

Base: gnn_GATv2Conv.py, con le correzioni seguenti:
  [CORREZIONE] Rimosso aggr='mean' da GATv2Conv: i pesi di attenzione (softmax) sommano
               già a 1 e vanno SOMMATI (aggregazione di default). Con 'mean' venivano
               divisi una seconda volta per il numero di vicini.
  [ATTENZIONE] HEADS teste di attenzione, mediate (concat=False): la dimensione di uscita
               resta layers_size, come per gli altri modelli.
  [MODIFICA 2] Il vettore dell'ULTIMO nodo del prefisso viene aggiunto al riassunto del grafo.
  [MODIFICA 4] LayerNorm dopo ogni layer e connessione residua a partire dal 2° layer.

Lo scheletro è identico a gnn_v2.py: cambia solo il tipo di layer (GATv2Conv invece di SAGEConv).
Il parametro 'aggregator' degli iperparametri non viene usato da questo modello.
"""
import torch
from torch.nn import Module, ModuleList, LayerNorm
from torch.nn.functional import relu, dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter

# Numero di teste di attenzione (1 = come il file originale)
HEADS = 4


class GATv2_GNN(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        # Layer GATv2Conv (hidden_layers + 2 layer, come negli altri modelli)
        # [CORREZIONE] nessun aggr='mean' ; [ATTENZIONE] HEADS teste mediate
        self.convs = ModuleList()
        self.convs.append(GATv2Conv(in_channels=self.input_size, out_channels=self.layers_size,
                                    heads=HEADS, concat=False))
        for _ in range(self.num_hidden):
            self.convs.append(GATv2Conv(in_channels=self.layers_size, out_channels=self.layers_size,
                                        heads=HEADS, concat=False))
        self.convs.append(GATv2Conv(in_channels=self.layers_size, out_channels=self.layers_size,
                                    heads=HEADS, concat=False))

        # [MODIFICA 4] una normalizzazione per ogni layer
        self.norms = ModuleList([LayerNorm(self.layers_size) for _ in range(len(self.convs))])

        # [MODIFICA 2] lin1 riceve il riassunto del grafo + il vettore dell'ultimo nodo (2 x layers_size)
        self.lin1 = Linear(2 * self.layers_size, self.layers_size)
        self.lin2 = Linear(self.layers_size, self.output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:  # un solo grafo (fuori dal DataLoader)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)                                                   # [MODIFICA 4] normalizzazione
            h = relu(h)
            h = dropout(h, p=self.dropout, training=self.training)
            x = h if i == 0 else x + h                                    # [MODIFICA 4] residuo

        graph = self.readout(x, batch)                                    # riassunto del grafo

        # [MODIFICA 2] ultimo nodo di ogni grafo del batch (indice più alto = attività più recente).
        # Il target (attività successiva) non è mai in x.
        node_idx = torch.arange(x.size(0), device=x.device)
        last_idx = scatter(node_idx, batch, dim=0, reduce='max')
        last = x[last_idx]

        x = torch.cat([graph, last], dim=-1)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        return out
