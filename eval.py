import torch
from torch._C import device
import torch.nn.functional as F

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
