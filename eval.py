import torch
from torch import tensor
from torch._C import device
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def check_rmse(model, test_loader, device):
    num_samples = 0
    sum_rmse = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            num_samples += 1
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            sum_rmse += ((z - y)**2)
        return torch.sqrt(sum_rmse / num_samples)

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
    num_samples = 0
    sum_y = 0
    sum_num = 0
    sum_denom = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            num_samples += 1
            y = y.to(device=device)
            sum_y += y
        avg_y = sum_y / num_samples

        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            z = model(x)
            sum_num += ((z - avg_y) ** 2)
            sum_denom += ((y - avg_y) ** 2)
        return sum_num / sum_denom

if __name__ == "__main__":
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_save_path = 'E:\School\Imperial\individual_project\individual_project\models\EmadiNINet_130521.pt'
    model = torch.load(model_save_path)
    model.eval()

    image = rasterio.open(f'E:\School\Imperial\individual_project\qgisdata\Multi-size Mosaic_resampled_9band_reprojected_clipped.tif')

    num_bands = image.count
    img_width = image.width
    img_height = image.height
    num_pixels = img_height * img_width
    all_data = []

    for i in range(num_bands):
        all_data.append(image.read(i+1))

    ndvi_array = (all_data[8] - all_data[5]) / (all_data[8] + all_data[5])
    evi_array = 2.5 * (all_data[8] - all_data[4]) / (all_data[8] + 6 * all_data[4] - 7.5 * all_data[2] + 1)
    satvi_array = 2 * (all_data[0] - all_data[4]) / (all_data[0] + all_data[4] + 1) - (all_data[1] / 2)

    all_data.append(ndvi_array)
    all_data.append(evi_array)
    all_data.append(satvi_array)

    tensor_data = torch.Tensor(all_data).to(device=device)
    tensor_data = tensor_data.transpose(2, 0).transpose(1, 0)
    print(tensor_data)
    # tensor_data = []

    print("Calculating SOC...")
    result_data = []
    with torch.no_grad():
        for t in tqdm(tensor_data):
            result_data.append(model(t).cpu())

    result_data = torch.stack(result_data).detach().numpy()

    plt.imshow(result_data, cmap='summer')
    plt.colorbar()
    plt.show()
    print(image.shape)
    print(image.count)