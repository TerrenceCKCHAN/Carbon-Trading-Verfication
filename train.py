import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.models import EmadiNINet
import eval

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

def add_indices_columns(data_df):
    '''
    Adds NDVI, EVI, SATVI vegetation indices to dataset
    '''
    data_df['NDVI'] = (data_df['BAND_8'] - data_df['BAND_4']) / (data_df['BAND_8'] + data_df['BAND_4'])
    data_df['EVI'] = 2.5 * (data_df['BAND_8'] - data_df['BAND_4']) / (data_df['BAND_8'] + 6 * data_df['BAND_4'] - 7.5 * data_df['BAND_2'] + 1)
    data_df['SATVI'] = 2 * (data_df['BAND_11'] - data_df['BAND_4']) / (data_df['BAND_11'] + data_df['BAND_4'] + 1) - (data_df['BAND_12'] / 2)
    return data_df

csv_file_path = "E:\School\Imperial\individual_project\individual_project\data\lucas_sentinel2_data_points_zhou2020.csv"
lr = 0.0005
epochs = 1000
data_df = load_csv_to_pd(csv_file_path)
data_df = add_indices_columns(data_df)

# Split into train and test sets (90/10)
msk = np.random.rand(len(data_df)) < 1
train_df = data_df[msk]
test_df = data_df[~msk]


train_labels_tensor = torch.tensor(train_df['OC'].values.astype(np.float32))
train_data_tensor = torch.tensor(train_df.drop(['POINT_ID','OC','sample_ID','latitude','longitude','BAND_12','BAND_3','BAND_4','BAND_5','BAND_6','BAND_8','BAND_8A','SATVI','EVI'], axis = 1).values.astype(np.float32)) 
train_tensor = TensorDataset(train_data_tensor, train_labels_tensor) 
train_loader = DataLoader(dataset=train_tensor, batch_size=1, shuffle=True)

test_labels_tensor = torch.tensor(train_df['OC'].values.astype(np.float32))
test_data_tensor = torch.tensor(train_df.drop(['POINT_ID','OC','sample_ID','latitude','longitude','BAND_12','BAND_3','BAND_4','BAND_5','BAND_6','BAND_8','BAND_8A','SATVI','EVI'], axis = 1).values.astype(np.float32)) 
test_tensor = TensorDataset(test_data_tensor, test_labels_tensor) 
test_loader = DataLoader(dataset=test_tensor, batch_size = 1)

model = EmadiNINet()
model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=lr)

plt.ion()
fig, ax = plt.subplots(2, figsize=(5, 10))
train_losses = []
test_rmses = []

for e in range(epochs):
    total_loss = 0
    total_t = 0
    for t, (x, y) in enumerate(train_loader):
        model.train()
        x = x.to(device=device)
        y = y.to(device=device)

        z = model(x)

        loss = F.mse_loss(z, y)
        total_t += 1
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    rmse = float(eval.check_rmse(model, test_loader, device))
    mae = float(eval.check_mae(model, test_loader, device))
    r2 = float(eval.check_r2(model, test_loader, device))
    total_loss = total_loss.cpu().item() / total_t
    
    train_losses.append(total_loss)
    test_rmses.append(rmse)
    ax[0].plot(train_losses, c='black')
    ax[0].set_title('Train loss')
    
    ax[1].plot(test_rmses, c='black')
    ax[1].set_title('Test RMSE')

    plt.pause(0.05)

    print('Epoch {:d} | Loss {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(e, total_loss, rmse, mae, r2))

fig.savefig('graph.png')
plt.close(fig)
plt.show()