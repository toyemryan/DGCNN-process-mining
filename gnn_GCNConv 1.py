from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu, dropout
from torch_geometric.nn.dense.linear import Linear


class GCN_GNN(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.improved = parameters.get('improved')

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        self.convs = ModuleList()

        self.convs.append(GCNConv(in_channels=self.input_size, out_channels=self.layers_size, improved = self.improved))

        for _ in range(self.num_hidden):
            self.convs.append(GCNConv(in_channels=self.layers_size, out_channels=self.layers_size, improved = self.improved))

        self.convs.append(GCNConv(in_channels=self.layers_size, out_channels=self.layers_size, improved = self.improved))

        self.lin1 = Linear(self.layers_size, self.layers_size)
        self.lin2 = Linear(self.layers_size, self.output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = relu(x)
            x = dropout(x, p=self.dropout, training=self.training)

        x = self.readout(x, batch)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        return out
