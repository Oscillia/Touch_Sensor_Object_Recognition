# imports
# import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import KFold
from Dataset_Generation import *
from datetime import datetime


# randomization
def set_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > (2 ** 32 - 1):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed(100000007)


# initialization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root_dir = 'dataset'
num_workers = 2
batch_size = 8
num_epoch = 15
num_classes= 5 # number of classes; tbc
fold_k = 5 # for k-fold
csv_file_path = 'object_raw_images.csv' # with raw image
model_save_dir = 'models'
loss_list, accr_list = [], []


dataset_csv_generator(size=500)


# transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([125 / 255, 124 / 255, 115 / 255],
                         [60 / 255, 59 / 255, 64 / 255])
])

transform_train = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(90),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([125 / 255, 124 / 255, 115 / 255],
                         [60 / 255, 59 / 255, 64 / 255])
])

# dataset

dataset = TouchSensorObjectsDataset(csv_file_path, root_dir=data_root_dir, transform=transform_train)
train_set, test_set = random_split(dataset, [1800, 700])
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=num_workers)


# model and critetion
model_resnet = models.resnet18()
model_resnet.fc = nn.Linear(512, num_classes)
model_resnet.to(device)
criterion = nn.CrossEntropyLoss()


# k-fold
splits = KFold(n_splits=fold_k, shuffle=True, random_state=42)


def train_epoch(model, device, dataloader, criterion, optimizer):

    train_loss, train_correct = 0.0, 0
    loss_list, accr_list = [], []
    num_item = 0
    model.train()

    for _, data in enumerate(tqdm(dataloader)):

        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*inputs.size(0)
        num_item += inputs.size(0)
        loss_list.append(train_loss / num_item)

        _, predictions = torch.max(outputs.data, 1)
        train_correct += (predictions == labels).sum().item()
        accr_list.append(100.0*train_correct/num_item)

    return loss_list, accr_list


def valid_epoch(model, device, dataloader, criterion):

    valid_loss, valid_correct = 0.0, 0
    loss_list, accr_list = [], []
    num_item = 0
    model.eval()

    for _, data in enumerate(tqdm(dataloader)):

        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)


        valid_loss += loss.item()*inputs.size(0)
        num_item += inputs.size(0)
        loss_list.append(valid_loss / num_item)

        _, predictions = torch.max(outputs.data, 1)
        valid_correct += (predictions == labels).sum().item()
        accr_list.append(100.0*valid_correct/num_item)

    return loss_list, accr_list


history = {'train_loss': [], 'valid_loss': [],'train_accr':[],'valid_accr':[]}


def train(model, num_epoch=5, learning_rate=1e-3, save_interval=5):

    global history

    optimizer = torch.optim.Adam(model_resnet.parameters(), lr=learning_rate)

    # model = model_resnet
    model.to(device)

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        print("Epoch {} for learning rate {}:".format(epoch+1, learning_rate))

        # k-fold
        for fold, (train_idx, valid_idx) in enumerate(splits.split(np.arange(len(train_set)))):

            print("Fold {}:".format(fold+1))
            train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
            train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, drop_last=False)
            valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler, drop_last=False)

            train_loss_list, train_accr_list = train_epoch(model,device,train_loader,criterion,optimizer)
            valid_loss_list, valid_accr_list = valid_epoch(model,device,valid_loader,criterion)

            history['train_loss'].extend(train_loss_list)
            history['valid_loss'].extend(valid_loss_list)
            history['train_accr'].extend(train_accr_list)
            history['valid_accr'].extend(valid_accr_list)

            print('Epoch {}, fold {}: avg train loss {}, avg valid loss {}, avg train accr {}, avg valid accr {}'.format(epoch + 1, fold + 1, history['train_loss'][-1], history['valid_loss'][-1], history['train_accr'][-1], history['valid_accr'][-1]))

        if (epoch+1)%save_interval == 0:
            torch.save(model, os.path.join(model_save_dir, 'model_resnet18_{}_{}_{}.pth'.format(learning_rate*1e6, epoch+1, datetime.now().strftime('%Y%m%d%H%M%S'))))

    return


train(model_resnet,num_epoch=5,learning_rate=5e-4,save_interval=5)
train(model_resnet,num_epoch=5,learning_rate=1e-4,save_interval=5)
train(model_resnet,num_epoch=5,learning_rate=3e-5,save_interval=5)


plt.plot(history['valid_accr'])
plt.xlabel("iterations")
plt.ylabel("Validation Accuracy")
plt.show()