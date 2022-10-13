import os
# import numpy as np
import random
import pandas as pd
import torch
# from torch.nn.functional import one_hot
from torch.utils.data import Dataset
# from skimage import io
from PIL import Image
import glob

object_types = (['sphere', 'cylinder', 'nut', 'bolt', 'spanner',
                'lego_1x2', 'lego_2x3',
                 'usb_mini', 'usb_typec',
                 'key', 'hex_key', 'mesh_pot'])  # tbc
# old_object_types = [sphere, cylinder, nut, bolt] # with size 500

dataset_describer_path = 'object_raw_images'  # tbc
root_dir = 'dataset'


def get_file_names(root_dir=root_dir, object_name='', prefix='raw'):
    object_path_name = os.path.join(root_dir, object_name, prefix)
    names = glob.glob(os.path.join(object_path_name, '*'))
    # print(object_path_name, names)
    return names


# split train and test here
def dataset_csv_generator(objects=object_types, size=360, prefix='raw', save_path=dataset_describer_path):
    # objects is the list of name of objects
    # size is the number of pictures within each object type
    # old_object_types = ['sphere', 'cylinder', 'nut', 'bolt', 'spanner']  # with size 500
    file_names, tags = [], []
    for i in range(len(objects)):
        object_name = objects[i]
        for j in range(size):
            tags.append(i)
        object_file_names = get_file_names(object_name=object_name, prefix=prefix)
        # print(len(object_file_names))
        object_file_names = random.sample(object_file_names, size)
        file_names.extend(object_file_names)
    dataset_describer = pd.DataFrame({'file_name': file_names, 'tag': tags})
    dataset_describer.to_csv(save_path+'.csv')
    trainset_describer = dataset_describer.sample(frac=0.7)
    trainset_describer.to_csv(save_path+'_train.csv')
    testset_describer = dataset_describer[~dataset_describer.index.isin(trainset_describer.index)]
    testset_describer.sample(frac=1.0)
    testset_describer.to_csv(save_path+'_test.csv')


class TouchSensorObjectsDataset(Dataset):
    def __init__(self, csv_file_path=dataset_describer_path, transform=None, num_classes = 5):
        self.annotations = pd.read_csv(csv_file_path, index_col=0)  # annotations[i, 0] is file name, annotations[i, 1] is tag
        # self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)  # number of images in the dataset

    def __getitem__(self, index):
        # if torch.is_tensor(index):
        #     index = index.tolist()

        # print(self.annotations.iloc[index, 0])
        # img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img_path = self.annotations.iloc[index, 0]
        img = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        # y_label = one_hot(y_label, self.num_classes)

        if self.transform:
            img = self.transform(img)

        return img, y_label


if __name__ == '__main__':
    dataset_csv_generator()