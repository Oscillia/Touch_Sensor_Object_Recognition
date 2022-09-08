import os
# import numpy as np
import random
import pandas as pd
import torch
# from torch.nn.functional import one_hot
from torch.utils.data import Dataset
# from skimage import io
from PIL import Image

object_types = ['sphere', 'cylinder', 'nut', 'bolt', 'spanner']  # tbc
# old_object_types = [sphere, cylinder, nut, bolt] # with size 500

dataset_describer_path = 'object_raw_images.csv'  # tbc


def get_file_names(object_name='', prefix='raw', size=500):
    names = []
    for i in range(size):
        name = os.path.join(object_name, prefix, prefix + '_' + str(i) + '.png')
        names.append(name)
    return names


def dataset_csv_generator(objects=object_types, size=100, prefix='raw', save_path=dataset_describer_path):
    # objects is the list of name of objects
    # size is the number of pictures within each object type
    old_object_types = ['sphere', 'cylinder', 'nut', 'bolt']  # with size 500
    file_names, tags = [], []
    for i in range(len(objects)):
        object_name = objects[i]
        if object_name in old_object_types:
            for j in range(500):
                tags.append(i)
            object_file_names = get_file_names(object_name=object_name, prefix=prefix, size=500)
            # object_file_names = random.sample(object_file_names, size)
        else:
            object_file_names = get_file_names(object_name=object_name, prefix=prefix, size=size)
            for j in range(size):
                tags.append(i)
        file_names.extend(object_file_names)
    dataset_describer = pd.DataFrame({'file_name': file_names, 'tag': tags})
    dataset_describer.sample(frac=1)
    dataset_describer.to_csv(save_path)


class TouchSensorObjectsDataset(Dataset):
    def __init__(self, csv_file_path=dataset_describer_path, root_dir='dataset', transform=None, num_classes = 5):
        self.annotations = pd.read_csv(csv_file_path, index_col=0)  # annotations[i, 0] is file name, annotations[i, 1] is tag
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)  # number of images in the dataset

    def __getitem__(self, index):
        # if torch.is_tensor(index):
        #     index = index.tolist()

        # print(self.annotations.iloc[index, 0])
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        # y_label = one_hot(y_label, self.num_classes)

        if self.transform:
            img = self.transform(img)

        return img, y_label


if __name__ == '__main__':
    dataset_csv_generator()