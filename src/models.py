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
        dropout_value = 0.5
        self.layers = nn.Sequential(
            nn.Linear(13, 50),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )
    
    def forward(self, x):
        z = self.layers(x)
        z = z.view(x.shape[0])
        return z