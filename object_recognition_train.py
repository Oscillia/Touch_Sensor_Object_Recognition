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
print(device)
data_root_dir = 'dataset'
num_workers = 2
num_classes= 12 # number of classes; tbc
fold_k = 4 # for k-fold
csv_file_path = 'object_raw_images' # with raw image
model_save_dir = 'models'
loss_list, accr_list = [], []
learning_rate = 2e-3

# commented in the final version, since dataset is fixed
# run this before every training process if dataset is subject to change
# dataset_csv_generator(size=360, save_path=csv_file_path)

# normalization
# IMPORTANT: should re-run dataset_normalization_generation.py after each change for dataset
# load mean and std
normalization_factors = torch.load('normalization_factors_object_raw_images.pt')
# denormalization_factors = torch.load('denormalization_factors_object_raw_images.pt')

# transforms
transform_test = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(normalization_factors['img_mean'], normalization_factors['img_std'])
])

# transform_test_grayscale = transforms.Compose([
#     transform_test,
#     transforms.Grayscale(),
#     transforms.Normalize(normalization_factors['grayscale_img_mean'], normalization_factors['grayscale_img_std'])
# ])

transform_train_base = transforms.Compose([
    transforms.Resize(250),
    transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(90),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transform_train_base,
    transforms.Normalize(normalization_factors['img_mean'], normalization_factors['img_std'])
])

# transform_train_grayscale = transforms.Compose([
#     transform_train_base,
#     transforms.Grayscale(),
#     transforms.Normalize(normalization_factors['grayscale_img_mean'], normalization_factors['grayscale_img_std'])
# ])

# dataset

train_set_base = TouchSensorObjectsDataset(csv_file_path+'_train.csv', transform=transform_train)
# train_set_gray = TouchSensorObjectsDataset(csv_file_path+'_train.csv', transform=transform_train_grayscale)

test_set_base = TouchSensorObjectsDataset(csv_file_path+'_test.csv', transform=transform_test)
# test_set_gray = TouchSensorObjectsDataset(csv_file_path+'_test.csv', transform=transform_test_grayscale)
test_loader = DataLoader(test_set_base, batch_size=100, shuffle=False, num_workers=num_workers)
# test_loader_gray = DataLoader(test_set_gray, batch_size=100, shuffle=False, num_workers=num_workers)


# models and critetion
# for raw image
model_resnet = models.resnet18()
model_resnet.fc = nn.Linear(512, num_classes)
model_resnet.to(device)

# for grayscale image
# model_resnet_gray = models.resnet18()
# model_resnet_gray.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model_resnet_gray.fc = nn.Linear(512, num_classes)
# model_resnet_gray.to(device)

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


def train(model, train_set, test_loader, batch_size = 32, lr=1e-3, step_size=5, decay=0.3, num_epoch=20, save_interval=5, save_prefix=''):

    history = {'train_loss': [], 'valid_loss': [],'train_accr':[],'valid_accr':[]}
    optimizer = torch.optim.Adam(model_resnet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)

    # model = model_resnet
    model.to(device)

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        print("Epoch {} for learning rate {}:".format(epoch+1, scheduler.get_last_lr()))

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

        print("Training completed for Epoch {}. Evaluating on test set...".format(epoch + 1))
        evaluation(model, test_loader, device, img_show=True)

        if (epoch+1)%save_interval == 0:
            torch.save(model, os.path.join(model_save_dir, 'model_resnet18{}_{}_{}_{}.pth'.format(save_prefix, int(scheduler.get_last_lr()[-1] * 1000000), epoch+1, datetime.now().strftime('%Y%m%d%H%M%S'))))

        plt.plot(range(len(history['train_accr'])), history['train_accr'], label='train accuracy')
        plt.plot(range(len(history['valid_accr'])), history['valid_accr'], label='validation accuracy')
        plt.xlabel("iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        scheduler.step()

    return


@torch.no_grad()
def evaluation(model, dataLoader, device, img_show=False):
    correct = 0
    total = 0
    model.eval()
    denormalization_factors = torch.load('denormalization_factors_object_raw_images.pt')
    denorm = transforms.Normalize(denormalization_factors['denorm_mean'], denormalization_factors['denorm_std'])
    with torch.no_grad():
        for data in dataLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if img_show and (predicted != labels).sum().item() > 1:
                wrong_idx = np.where((predicted != labels).cpu() != 0).reshape(1, -1)
                # import pdb; pdb.set_trace()
                for idx in wrong_idx:
                    example = np.transpose(denorm(images[idx]).cpu().numpy(), (1, 2, 0)) # use this if input images are normalized
                    # example = np.transpose(images[idx].cpu().numpy(), (1, 2, 0))
                    plt.imshow(example.astype(np.uint8))
                    plt.title("predicted {}, should be {}".format(object_types[predicted[idx]], object_types[labels[idx]]))
                    plt.show()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {}%'.format(accuracy))
    return accuracy


train(model_resnet, train_set_base, test_loader, num_epoch=30)

evaluation(model_resnet, test_loader, device, img_show=True)

train(model_resnet, train_set_base, test_loader, lr=2e-3, step_size=3, decay=0.6, num_epoch=30)

evaluation(model_resnet, test_loader, device, img_show=True)


# train(model_resnet_gray, train_set_gray, test_loader_gray, lr=2e-3, step_size=3, decay=0.9, num_epoch=30, save_prefix='_grayscale')

# evaluation(model_resnet_gray, test_loader_gray, device, img_show=True)

