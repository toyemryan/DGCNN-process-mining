"""
Versione modificata di gnn.py (il file originale del professore non viene toccato).

Modifiche rispetto a gnn.py:
  [MODIFICA 2] Il vettore dell'ULTIMO nodo del prefisso (l'attività più recente) viene
               aggiunto al riassunto del grafo prima della classificazione.
  [MODIFICA 4] Una normalizzazione (LayerNorm) dopo ogni layer SAGEConv e una
               connessione residua a partire dal 2° layer.

Il resto (layer, readout, dropout, layer lineari finali) è identico a gnn.py.
"""
import torch
from torch.nn import Module, ModuleList, LayerNorm
from torch.nn.functional import relu, dropout
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter


class SAGE_GNN(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.aggregator = parameters.get('aggregator')

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        # Layer SAGEConv: identici a gnn.py (hidden_layers + 2 layer)
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels=self.input_size, out_channels=self.layers_size, aggr=self.aggregator))
        for _ in range(self.num_hidden):
            self.convs.append(SAGEConv(in_channels=self.layers_size, out_channels=self.layers_size, aggr=self.aggregator))
        self.convs.append(SAGEConv(in_channels=self.layers_size, out_channels=self.layers_size, aggr=self.aggregator))

        # [MODIFICA 4] una normalizzazione per ogni layer SAGEConv
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
            x = h if i == 0 else x + h                                    # [MODIFICA 4] residuo (dimensioni uguali dal 2° layer)

        graph = self.readout(x, batch)                                    # riassunto del grafo (come prima)

        # [MODIFICA 2] indice dell'ultimo nodo di OGNI grafo del batch = l'indice più alto di quel grafo.
        # I nodi sono ordinati cronologicamente da data_encoding.py,
        # quindi è l'ultima attività del prefisso. Il target (attività successiva) non è mai in x.
        node_idx = torch.arange(x.size(0), device=x.device)
        last_idx = scatter(node_idx, batch, dim=0, reduce='max')
        last = x[last_idx]

        x = torch.cat([graph, last], dim=-1)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        return out
