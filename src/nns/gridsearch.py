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
import itertools

from models import SimpleNet
import eval

layers_space = [2, 3, 5, 7]
neurons_space = [20, 50, 100]
dropout_space = [0.2, 0.5, 0.7]
lr_space = [0.01, 0.005, 0.001, 0.0005]
epochs_space = [20, 50, 100]
# layers_space = [7]
# neurons_space = [20]
# dropout_space = [0.2]
# lr_space = [0.05]
# epochs_space = [20]

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
data_df = load_csv_to_pd(csv_file_path)

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
]

idx = 0

results_df = pd.DataFrame(columns=['idx', 'lr', 'epochs', 'dropout', 'layers', 'neurons', 'test_rmse', 'test_mae', 'test_r2'])


for lr, epochs, dropout, layers, neurons in tqdm(list(itertools.product(lr_space, epochs_space, dropout_space, layers_space, neurons_space))):
    msk = np.random.rand(len(data_df)) < 0.5
    data_df_1 = data_df[msk]
    data_df_2 = data_df[~msk]
    rmses = []
    maes = []
    r2s = []
    for i in range(2):
        train_df = data_df_1
        test_df = data_df_2
        if i == 1:
            train_df = data_df_2
            test_df = data_df_1
        
        train_labels_tensor = torch.tensor(np.log(train_df['OC'].values.astype(np.float32)))
        train_data_tensor = torch.tensor(train_df[features_list].values.astype(np.float32)) 
        train_tensor = TensorDataset(train_data_tensor, train_labels_tensor) 
        train_loader = DataLoader(dataset=train_tensor, batch_size=16, shuffle=True)

        test_labels_tensor = torch.tensor(np.log(test_df['OC'].values.astype(np.float32)))
        test_data_tensor = torch.tensor(test_df[features_list].values.astype(np.float32)) 
        test_tensor = TensorDataset(test_data_tensor, test_labels_tensor) 
        test_loader = DataLoader(dataset=test_tensor, batch_size = 1)

        model = SimpleNet(len(features_list), layers, neurons, dropout)
        model = model.to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for e in range(epochs):
            total_t = 0
            for t, (x, y) in enumerate(train_loader):
                model.train()
                x = x.to(device=device)
                y = y.to(device=device)

                z = model(x)
                z_array = z.detach().cpu().numpy()

                loss = F.mse_loss(z, y)
                total_t += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        rmse = float(eval.check_rmse(model, test_loader, device))
        mae = float(eval.check_mae(model, test_loader, device))
        r2 = float(eval.check_r2(model, test_loader, device))
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
    total_rmse = sum(rmses) / len(rmses)
    total_mae = sum(maes) / len(maes)
    total_r2 = sum(r2s) / len(r2s)

    result_dict = {}
    result_dict['idx'] = idx
    result_dict['lr'] = lr
    result_dict['epochs'] = epochs
    result_dict['dropout'] = dropout
    result_dict['layers'] = layers
    result_dict['neurons'] = neurons
    result_dict['test_rmse'] = total_rmse
    result_dict['test_mae'] = total_mae
    result_dict['test_r2'] = total_r2

    results_df = results_df.append(result_dict, ignore_index=True)
    print('IDX: {:d} | lr {:.4f} | epochs {:d} | dropout {:.4f} | layers {:d} | neurons {:d} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(idx, lr, epochs, dropout, layers, neurons, total_rmse, total_mae, total_r2))

    idx += 1

results_df.to_csv('out/nn_gridsearch.csv', index=False)
