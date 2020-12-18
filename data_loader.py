from torch.utils.data import Dataset


class DatasetReader(Dataset):
    def __init__(self, samples, labels, transforms=None):
        self.samples = samples
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'x': self.samples[item], 'y': self.labels[item]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
