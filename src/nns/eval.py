import torch
from torch import tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

def check_rmse(model, test_loader, device):
    num_samples = 0
    sum_rmse = 0
    model.eval()
    with torch.no_grad():
        ys = []
        zs = []
        for x, y in test_loader:
            num_samples += 1
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            ys.append(y.detach().cpu().item())
            zs.append(z.detach().cpu().item())
        return np.sqrt(mean_squared_error(ys, zs))

def check_rmspe(model, test_loader, device):
    num_samples = 0
    sum_rmse = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            num_samples += 1
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            sum_rmse += (((z - y) / y)**2)
        return torch.sqrt(sum_rmse / num_samples)

def check_mae(model, test_loader, device):
    num_samples = 0
    sum_mae = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            num_samples += 1
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            sum_mae += torch.abs(z - y)
        return sum_mae / num_samples

def check_mape(model, test_loader, device):
    num_samples = 0
    sum_mae = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            num_samples += 1
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            sum_mae += torch.abs((z - y) / y)
        return sum_mae / num_samples

def check_r2(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        ys = []
        zs = []
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            ys.append(y.detach().cpu().item())
            zs.append(z.detach().cpu().item())
        return r2_score(ys, zs)

if __name__ == "__main__":
    USE_GPU = True
    print("Start")
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using:", device)
    print("Loading Model...")
    model_save_path = r'C:\Users\kothi\Documents\individual_project\individual_project\models\model.pt'
    model = torch.load(model_save_path)
    model.eval()

    print("Loading Raster...")
    image = rasterio.open(fr'C:\Users\kothi\Documents\individual_project\qgisdata\S1AIW_S2AL2A_NDVI_SATVI_EVI_DEM.tif')
    # image = rasterio.open(fr'C:\Users\kothi\Documents\individual_project\qgisdata\S2A1C_DEM.tif')

    # num_bands = 10
    num_bands = image.count
    img_width = image.width
    img_height = image.height
    num_pixels = img_height * img_width
    all_data = []
    print("Image shape:", image.shape)


    print("Converting Raster to Array...")
    for i in tqdm(range(num_bands)):
        data = image.read(i+1)
        data = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        all_data.append(data)

    all_data = np.dstack(all_data)
    all_data_shape = all_data.shape
    print("Raster array shape:", all_data_shape)

    print(np.any(np.isnan(all_data))) # False
    print(np.all(np.isfinite(all_data))) # True

    print("Calculating SOC...")
    result_data = []
    non_zero = 0
    with torch.no_grad():
        for t in tqdm(all_data):
            t = torch.Tensor(t).to(device=device)
            z = model(t).cpu()
            result_data.append(z)
            non_zero += torch.count_nonzero(z)
            z[z!=z] = 0
    print("non_zero:", non_zero)

    result_data = np.exp(torch.stack(result_data).detach().numpy())
    print("max val: ", np.max(result_data))
    plt.hist(result_data.flatten(), bins=np.linspace(0, 500, 100), histtype=u'step', density=True)
    plt.savefig('nn_inference_histogram.png')
    plt.show()

    plt.imshow(result_data, cmap='viridis_r')
    plt.colorbar()
    plt.savefig('nn_map.png')
    plt.show()

    with rasterio.open(
        'out/nn_map.tif',
        'w',
        driver='GTiff',
        height=result_data.shape[0],
        width=result_data.shape[1],
        count=1,
        dtype=result_data.dtype,
        crs='+proj=latlong',
        transform=image.transform,
    ) as dst:
        dst.write(result_data, 1)