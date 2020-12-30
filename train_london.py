import time

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from source.data_loader import LondonDatasetReader
from source.utils import get_household_complete_data, train_val_test_separator, ToTensor

#####################################################################
# Defining dataset paths
dataset_paths = {
    'household_info': 'datasets/London/informations_households.csv',
    'acorn_groups': 'datasets/London/acorn_details.csv',
    'weather_daily': 'datasets/London/weather_daily_darksky.csv',
    'weather_hourly': 'datasets/London/weather_hourly_darksky.csv',
    'holidays': 'datasets/London/uk_bank_holidays.csv',
    'daily_block': 'datasets/London/daily_dataset/daily_dataset/',
    'hh_block': 'datasets/London/halfhourly_dataset/halfhourly_dataset/'
}
#####################################################################
# Reading the target half-hourly dataset
print('[*] Generating the dataset...')
available_blocks = [f'block_{i}' for i in range(112)]
target_block = available_blocks[0]

dataframe = pd.read_csv(dataset_paths['hh_block'] + target_block + '.csv')

available_household_ids = list(np.unique(dataframe['LCLid']))

target_households = [available_household_ids[0], available_household_ids[1], available_household_ids[2]]

samples = []
labels = []

for household in target_households:
    x, y = get_household_complete_data(dataframe, household)
    samples.append(x)
    labels.append(y)

train_samples = []
train_labels = []

val_samples = []
val_labels = []

test_samples = []
test_labels = []

for i in range(len(samples)):
    separated_data = train_val_test_separator(samples[i], labels[i])
    train_samples.append(separated_data[0])
    train_labels.append(separated_data[1])
    val_samples.append(separated_data[2])
    val_labels.append(separated_data[3])
    test_samples.append(separated_data[4])
    test_labels.append(separated_data[5])

#####################################################################
# Generating PyTorch friendly dataset

train_dataset = LondonDatasetReader(train_samples, train_labels, transforms=Compose([ToTensor()]))
val_dataset = LondonDatasetReader(val_samples, val_labels, transforms=Compose([ToTensor()]))
test_dataset = LondonDatasetReader(test_samples, test_samples, transforms=Compose([ToTensor()]))

#####################################################################
# Setting the global parameters

train_batch_size = 10
val_batch_size = 10
epochs = 1

#####################################################################
# Loading the data

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#####################################################################
# Building the model, loss function and optimizer

for i, sample in enumerate(train_loader):
    print(sample['x'].size(), sample['y'].size())
    exit()




