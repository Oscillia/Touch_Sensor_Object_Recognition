import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from Dataset_Generation import *


transform_not_normalized = transforms.ToTensor()

transform_gray_not_normalized = transforms.Compose([
    transform_not_normalized,
    transforms.Grayscale(),
])



csv_file_name = 'object_raw_images' # with raw image
dataset = TouchSensorObjectsDataset(csv_file_path=csv_file_name+'.csv', transform=transform_not_normalized)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
imgs = torch.stack([img_t for img_t, _ in dataset], dim=3)
img_mean, img_std = imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1)

# dataset_gray = TouchSensorObjectsDataset(csv_file_path=csv_file_name+'.csv', transform=transform_gray_not_normalized)
# imgs_gray = torch.stack([img_t for img_t, _ in dataset], dim=3)
# img_mean_gray, img_std_gray = imgs.view(1, -1).mean(dim=1), imgs.view(1, -1).std(dim=1)

normalization_factors = {'img_mean': img_mean, 'img_std': img_std}
# normalization_factors = {'img_mean': img_mean, 'img_std': img_std, 'grayscale_img_mean': img_mean_gray, 'grayscale_img_std': img_std_gray}
torch.save(normalization_factors, 'normalization_factors_'+csv_file_name+'.pt')

# print(1)

# denormalization
denorm_mean, denorm_std = - img_mean / img_std, 1 / img_std
# denorm_mean_gray, denorm_std_gray = - img_mean_gray / img_std_gray, 1 / img_std_gray

denormalization_factors = {'denorm_mean': img_mean, 'denorm_std': img_std}
torch.save(denormalization_factors, 'denormalization_factors_'+csv_file_name+'.pt')

# print('end')