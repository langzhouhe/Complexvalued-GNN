import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import *
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir='log')

class CGNN(nn.Module):
    def __init__(self, num_features, num_layers, num_hidden, num_class, dropout, labels, num_nodes, device):
        super(CGNN, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(AngularAggLayer(num_nodes, num_class, num_hidden, labels, dropout, device, name=i))
        self.fcs = nn.ModuleList()
        self.fcs.append(ComplexLinear(num_features, num_hidden, 0))
        self.fcs.append(ComplexLinear(num_hidden, num_class, 1))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act = nn.ReLU()
        self.dropout = dropout

    def complex_relu(self, input):
        return F.relu(input.real).type(torch.complex64)+1j * F.relu(input.imag).type(torch.complex64)

    def forward(self, x, adj):
        _layers = []
        layer_inner = self.complex_relu(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            if i == 0:
                layer_inner = self.complex_relu((con(layer_inner, adj, True)))
            else:
                layer_inner = self.complex_relu(con(layer_inner, adj))

        layer_inner = self.fcs[-1](layer_inner)
        layer_inner = torch.angle(layer_inner)
        return F.log_softmax(layer_inner, dim=1)

    # def __del__(self):
    #     writer.close()

if __name__ == '__main__':
    pass