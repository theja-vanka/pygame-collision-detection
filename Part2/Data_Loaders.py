import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',', dtype=np.float32)
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        self.n_samples = self.normalized_data.shape[0]
        self.n_features = self.normalized_data.shape[1]
        minority = self.normalized_data[np.where(self.normalized_data[:, 6] == 1)]
        majority = self.normalized_data[np.where(self.normalized_data[:, 6] == 0)]
        minority_upsampled = resample(minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=majority.shape[0],    # to match majority class
                                 random_state=42)
        self.dset = np.concatenate((majority, minority_upsampled), axis=0)
        self.n_samples = self.dset.shape[0]
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
    # STUDENTS: __len__() returns the length of the dataset
        return self.n_samples

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        x = torch.from_numpy(self.dset[idx][0:self.n_features-1])
        y = torch.tensor(self.dset[idx][self.n_features-1], dtype=torch.float32)
        return {
            'input': x,
            'label': y,
        }
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        self.train_loader = []
        self.test_loader = []
        X_train, X_test = train_test_split(self.nav_dataset, test_size=0.2)
        for idx, value in enumerate(X_train):
            self.train_loader.append(value)
        
        for idx, value in enumerate(X_test):
            self.test_loader.append(value)
        

# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
