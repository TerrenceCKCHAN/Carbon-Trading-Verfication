import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import SimpleNet
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

csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.04.csv"
lucas_csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_LUCAS2009_zhou2020_points.csv"
# csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S2A1C_DEM_LUCASTIN_roi_points.csv"
lr = 0.001
epochs = 100
data_df = load_csv_to_pd(csv_file_path)
lucas_data_df = load_csv_to_pd(lucas_csv_file_path)

msk = np.random.rand(len(data_df)) < 0.8
train_df = data_df[msk]
test_df = data_df[~msk]

print("Length of training set: ", len(train_df))
print("Length of test set: ", len(test_df))

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
]

# features_list = [
#     'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI'
# ]

train_labels_tensor = torch.tensor(np.log(train_df['OC'].values.astype(np.float32)))
train_data_tensor = torch.tensor(train_df[features_list].values.astype(np.float32)) 
train_tensor = TensorDataset(train_data_tensor, train_labels_tensor) 
train_loader = DataLoader(dataset=train_tensor, batch_size=32, shuffle=True)

test_labels_tensor = torch.tensor(np.log(test_df['OC'].values.astype(np.float32)))
test_data_tensor = torch.tensor(test_df[features_list].values.astype(np.float32)) 
test_tensor = TensorDataset(test_data_tensor, test_labels_tensor) 
test_loader = DataLoader(dataset=test_tensor, batch_size = 1)

model = SimpleNet(len(features_list), layers=7, neurons=100, dropout=0.2)
model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=lr)

plt.ion()
fig, ax = plt.subplots(2, 3, figsize=(10, 5))
train_losses = []
test_rmses = []
test_maes = []
test_r2s = []

for e in tqdm(range(epochs)):
    total_loss = 0
    total_t = 0
    zs = []
    for t, (x, y) in enumerate(train_loader):
        model.train()
        x = x.to(device=device)
        y = y.to(device=device)

        z = model(x)
        z_array = z.detach().cpu().numpy()
        zs.extend(z_array)

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
    test_maes.append(mae)
    test_r2s.append(r2)
    
    ax[0,0].plot(train_losses, c='black')
    ax[0,0].set_title('Train loss')
    
    ax[1,0].plot(test_rmses, c='black')
    ax[1,0].set_title('Test RMSE')
    
    ax[0,1].plot(test_maes, c='black')
    ax[0,1].set_title('Test MAE')
    
    ax[1,1].plot(test_r2s, c='black')
    ax[1,1].set_title('Test R^2')
    
    ax[0,2].clear()
    ax[1,2].clear()
    ax[0,2].hist(train_df['OC'].values.astype(np.float32), bins=np.linspace(0, 500, 100), histtype=u'step', density=True)
    ax[1,2].hist(np.exp(zs), bins=np.linspace(0, 500, 100), histtype=u'step', density=True)
    plt.pause(0.05)

    print('Epoch {:d} | Loss {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(e, total_loss, rmse, mae, r2))

lucas_labels_tensor = torch.tensor(np.log(lucas_data_df['OC'].values.astype(np.float32)))
lucas_data_tensor = torch.tensor(lucas_data_df[features_list].values.astype(np.float32)) 
lucas_tensor = TensorDataset(lucas_data_tensor, lucas_labels_tensor) 
lucas_loader = DataLoader(dataset=lucas_tensor, batch_size=1)

lucas_rmse = float(eval.check_rmse(model, lucas_loader, device))
lucas_mae = float(eval.check_mae(model, lucas_loader, device))
lucas_r2 = float(eval.check_r2(model, lucas_loader, device))
print('LUCAS2009 ZHOU2020 RESULTS: | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(lucas_rmse, lucas_mae, lucas_r2))


fig.savefig('nn_training_graph.png')
plt.close(fig)
plt.show()
plt.ioff()
model_save_path = 'models/model.pt'

torch.save(model, model_save_path)