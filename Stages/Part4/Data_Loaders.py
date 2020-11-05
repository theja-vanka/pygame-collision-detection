import torch
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from torch.utils.data.sampler import SubsetRandomSampler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.dset = np.genfromtxt('saved/training_data.csv', delimiter=',', dtype=np.float32)
# it may be helpful for the final part to balance the distribution of your collected data
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.dset) #fits and transforms
        self.n_samples = self.normalized_data.shape[0]
        self.n_features = self.normalized_data.shape[1]
        minority = self.normalized_data[np.where(self.normalized_data[:, 6] == 1)]
        majority = self.normalized_data[np.where(self.normalized_data[:, 6] == 0)]
        minority_upsampled = resample(
                                minority,
                                replace=True,     # sample with replacement
                                n_samples=majority.shape[0],    # to match majority class
                                random_state=42
                            )
        self.data = np.concatenate((minority_upsampled, majority), axis=0)
        self.n_samples = self.data.shape[0]
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        # __len__() returns the length of the dataset
        return self.n_samples

    def __getitem__(self, idx):
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        device = torch.device(dev)
        if not isinstance(idx, int):
            idx = idx.item()
        x = torch.from_numpy(self.data[idx][0:self.n_features-1]).to(device)
        y = torch.tensor(self.data[idx][self.n_features-1], dtype=torch.float32).to(device)
        return {
            'input': x,
            'label': y,
        }
# for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32.
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        shuffle_dataset = True
        random_seed = 42
        dataset_size = self.nav_dataset.n_samples
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(
                self.nav_dataset,
                batch_size=batch_size,
                sampler=train_sampler
            )
        self.test_loader = torch.utils.data.DataLoader(
                self.nav_dataset,
                batch_size=batch_size,
                sampler=valid_sampler
            )

# randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']


if __name__ == '__main__':
    main()
