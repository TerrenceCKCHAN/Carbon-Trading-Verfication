import numpy as np
import pandas as pd
import joblib
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

print("Start")
model_path = "models/brtmodel.joblib.pkl"
print("Loading model", model_path, "...")
brt = joblib.load(model_path)

print("Loading Raster...")
image = rasterio.open(fr'C:\Users\kothi\Documents\individual_project\qgisdata\S1AIW_S2AL2A_NDVI_SATVI_EVI_DEM.tif')

num_bands = image.count
img_width = image.width
img_height = image.height
num_pixels = img_height * img_width
all_data = []
print("Image shape:", image.shape)

print("Converting Raster to Array...")
for i in tqdm(range(num_bands)):
    data = image.read(i+1)
    data = pd.DataFrame(data).fillna(0).to_numpy()
    all_data.append(data)

all_data = np.dstack(all_data)
all_data_shape = all_data.shape
print("Raster array shape:", all_data_shape)

print(np.any(np.isnan(all_data)))
print(np.all(np.isfinite(all_data)))

print("Calculating SOC...")
result_data = []
non_zero = 0
for t in tqdm(all_data):
    z = brt.predict(t)
    result_data.append(z)
    non_zero += np.count_nonzero(z)
    z[z!=z] = 0
    # print(torch.min(z), torch.max(z))
print("non_zero:", non_zero)

result_data = np.exp(np.stack(result_data))
print("max val: ", np.max(result_data))
plt.hist(result_data.flatten(), bins=np.linspace(0, 500, 100), histtype=u'step', density=True)
plt.savefig('brt_inference_histogram.png')
plt.show()
result_data[result_data > 200] = 200

plt.imshow(result_data, cmap='viridis_r')
plt.colorbar()
plt.savefig('brt_map.png')
plt.show()