import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    '''
    Network Architecture from the Emadi 2020 paper on Norther Iran Soils
    Estimated values:
        Dropout = 0.2-0.8
        Learning rate = 0.001-0.05
        Epochs = 100
    '''
    def __init__(self, input_neurons, layers=5, neurons=50, dropout=0.5):
        super(SimpleNet, self).__init__()
        dropout_value = dropout
        num_neurons = neurons
        layers_list = []
        assert(layers >= 2)
        layers_list.append(nn.Linear(input_neurons, num_neurons))
        layers_list.append(nn.Dropout(dropout_value))
        layers_list.append(nn.ReLU())

        for i in range(layers - 2):
            layers_list.append(nn.Linear(num_neurons, num_neurons))
            layers_list.append(nn.Dropout(dropout_value))
            layers_list.append(nn.ReLU())

        layers_list.append(nn.Linear(num_neurons, 1))
        layers_list.append(nn.ReLU())
        self.layers = nn.Sequential(*layers_list)
    
    def forward(self, x):
        z = self.layers(x)
        z = z.view(x.shape[0])
        return z