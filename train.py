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
lr = 0.01
epochs = 100
data_df = load_csv_to_pd(csv_file_path)
data_df = add_indices_columns(data_df)

# Split into train and test sets (90/10)
msk = np.random.rand(len(data_df)) < 1.0
train_df = data_df[msk]
test_df = data_df[~msk]


train_labels_tensor = torch.tensor(train_df['OC'].values.astype(np.float32))
train_data_tensor = torch.tensor(train_df.drop(['POINT_ID','OC','sample_ID','latitude','longitude'], axis = 1).values.astype(np.float32)) 
train_tensor = TensorDataset(train_data_tensor, train_labels_tensor) 
train_loader = DataLoader(dataset=train_tensor, batch_size=1, shuffle=True)

test_labels_tensor = torch.tensor(train_df['OC'].values.astype(np.float32))
test_data_tensor = torch.tensor(train_df.drop(['POINT_ID','OC','sample_ID','latitude','longitude'], axis = 1).values.astype(np.float32)) 
test_tensor = TensorDataset(test_data_tensor, test_labels_tensor) 
test_loader = DataLoader(dataset=test_tensor, batch_size = 1)

model = EmadiNINet()
model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=lr)

plt.ion()
fig, ax = plt.subplots(2, 3, figsize=(10, 5))
train_losses = []
test_rmspes = []
test_mapes = []
test_r2s = []

for e in range(epochs):
    total_loss = 0
    total_t = 0
    zs = []
    for t, (x, y) in enumerate(train_loader):
        model.train()
        x = x.to(device=device)
        y = y.to(device=device)

        z = model(x)
        zs.append(z.cpu().item())

        loss = F.mse_loss(z, y)
        total_t += 1
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    rmspe = float(eval.check_rmspe(model, test_loader, device))
    mape = float(eval.check_mape(model, test_loader, device))
    r2 = float(eval.check_r2(model, test_loader, device))
    total_loss = total_loss.cpu().item() / total_t
    
    train_losses.append(total_loss)
    test_rmspes.append(rmspe)
    test_mapes.append(mape)
    test_r2s.append(r2)
    
    ax[0,0].plot(train_losses, c='black')
    ax[0,0].set_title('Train loss')
    
    ax[1,0].plot(test_rmspes, c='black')
    ax[1,0].set_title('Test RMSPE')
    
    ax[0,1].plot(test_mapes, c='black')
    ax[0,1].set_title('Test MAPE')
    
    ax[1,1].plot(test_r2s, c='black')
    ax[1,1].set_title('Test R^2')
    
    ax[0,2].clear()
    ax[1,2].clear()
    ax[0,2].hist(train_df['OC'].values.astype(np.float32), bins=np.linspace(0, 500, 100), histtype=u'step', density=True)
    ax[1,2].hist(zs, bins=np.linspace(0, 500, 100), histtype=u'step', density=True)
    plt.pause(0.05)

    print('Epoch {:d} | Loss {:.4f} | RMSPE {:.4f} | MAPE {:.4f} | R2 {:.4f}'.format(e, total_loss, rmspe, mape, r2))

fig.savefig('graph.png')
plt.close(fig)
plt.show()
plt.ioff()
model_save_path = 'models/model.pt'

torch.save(model, model_save_path)