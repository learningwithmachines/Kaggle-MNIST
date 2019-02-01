#%% Change working directory from the workspace root to the ipynb file location.
# Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
import sys
import time
from IPython.core import getipython
from matplotlib import pyplot as plt
import seaborn as sns
import pickle as pkl
import multiprocessing as mp
import numpy as np
import pandas as pd
import PIL
from collections import OrderedDict as ODict
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import functional as fu
import torch as tc
import torchvision as tcvis

try:
    os.chdir(os.path.join(os.getcwd(), '..'))
    print(os.getcwd())
except NotADirectoryError:
    pass
getipython
sns.set(style='white', context='notebook', palette='tab10')
# %%
datadir = os.getcwd() + '/data'
filenames = ['train.csv', 'test.csv']
datadict = ODict()
for files in filenames:
    try:
        with open(datadir + '/' + files, mode='r') as csvfile:
            datadict[files] = np.loadtxt(csvfile, delimiter=",", skiprows=1)
            csvfile.close()
        print('found file: {}'.format(files))
    except FileNotFoundError:
        print('skipping file ./{}'.format(files))

datadict.keys(), filenames
# %%
traindata = datadict[filenames[0]]
testdata = datadict[filenames[-1]]

trainlabels = traindata[:, 0].reshape(-1, 1)
traindata = traindata[:, 1:].reshape(-1, 28, 28)
testdata = testdata.reshape(-1, 28, 28)
print(traindata.shape, trainlabels.shape, testdata.shape)

# %%
fig, ax = plt.subplots(1, 2, sharex=True, squeeze=True)
ax[0].imshow(traindata[-1, :, :], cmap='gray')
ax[1].imshow(testdata[0, :, :], cmap='gray')


# %%
class NpDataset(Dataset):

    def __init__(self, x=traindata, y=trainlabels,
                 transforms=None, target_transforms=None):
        super().__init__()

        self.x = x
        self.y = y
        self.transform = transforms
        self.target_transform = target_transforms

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, index):

        image, label = self.x[index], self.y[index]
        # HxWxC, UINT8
        image = image.astype(np.uint8).reshape(28, 28, 1)
        print(type(image))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


# test
test_set = NpDataset()
plt.imshow(test_set.__getitem__(0)[0].reshape(28, 28), cmap='gray')

# %%
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.25),
    transforms.RandomVerticalFlip(0.25),
    transforms.RandomRotation(30, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([transforms.ToTensor()])

# test
test_set = NpDataset(transforms=transform, target_transforms=target_transform)
test_set.__getitem__(0)[0]

# %%
