import torch
import random
import numpy as np


def series_to_sequence(dataset, target_column, input_lag, output_lag, remove_columns=None):
    x_columns = list(dataset.columns)
    x_columns.remove(target_column)

    if remove_columns is not None:
        for column in remove_columns:
            x_columns.remove(column)

    x = dataset[x_columns].values
    y = dataset[target_column].values

    ds_length = dataset.shape[0]

    new_samples = []
    new_labels = []

    for i in range(0, ds_length - (input_lag + output_lag)):
        new_sample = x[i: i + input_lag]
        new_label = y[i + input_lag: i + input_lag + output_lag]

        new_samples.append(new_sample)
        new_labels.append(new_label)

    return np.array(new_samples, dtype=np.float), np.array(new_labels, np.float)


def train_val_test_separator(samples, labels, train_ratio=0.8, val_ratio=0.2):
    ds_length = samples.shape[0]
    indices = [i for i in range(ds_length)]

    train_val_length = int(train_ratio * ds_length)

    train_val_indices = random.choices(indices, k=train_val_length)
    test_indices = sorted(list(set(indices).difference(set(train_val_indices))))

    val_indices = random.choices(train_val_indices, k=int(val_ratio * len(train_val_indices)))
    train_indices = sorted(list(set(train_val_indices).difference(set(val_indices))))

    train_samples, train_labels = samples[train_indices], labels[train_indices]
    val_samples, val_labels = samples[val_indices], labels[val_indices]
    test_samples, test_labels = samples[test_indices], labels[test_indices]

    return train_samples, train_labels, val_samples, val_labels, test_samples, test_labels


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        new_sample = {'x': torch.tensor(x, dtype=torch.float32), 'y': torch.tensor(y, dtype=torch.float32)}

        return new_sample


