import torch
import torch.nn as nn

class EmadiNINet(nn.Module):
    '''
    Network Architecture from the Emadi 2020 paper on Norther Iran Soils
    Estimated values:
        Dropout = 0.2-0.8
        Learning rate = 0.001-0.05
        Epochs = 100
    '''
    def __init__(self):
        super(EmadiNINet, self).__init__()
        dropout_value = 0.8
        num_neurons = 50
        self.layers = nn.Sequential(
            nn.Linear(13, num_neurons),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(num_neurons, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        z = self.layers(x)
        z = z.view(x.shape[0])
        return z