import os
import pickle
from os import listdir
from typing import List, Any

import torchvision
from PIL import Image as PImage
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import ImageFolder, default_loader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torchvision import models
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn import preprocessing
from utils import *
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import feature
from spectralloader import Spectralloader
from cnn import train
from explainer import *
from plots import *
from helpfunctions import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
retrain = True
plot_for_image_id, plot_classes = False, False
roar_create_mask = False
roar_train = True
N_EPOCHS = 120
lr = 0.00015


# cuda:1
#
# returns Array of tuples(String, int) with ID and disease information 0 disease/ 1 healthy e.g. (3_Z2_1_0_1, 0)
def load_labels():
    path_test = 'data/test_fileids.txt'
    path_train = 'data/train_fileids.txt'
    valid = []
    train = []
    valid_s = open(path_test, 'r').readlines()
    train_s = open(path_train, 'r').readlines()
    for i in valid_s:
        data = i.split(';')
        valid.append((data[0], int(data[1])))
    for i in train_s:
        data = i.split(';')
        train.append((data[0], int(data[1])))
    return train, valid, train + valid


def main():
    mode = 'rgb'
    shuffle_dataset = True
    random_seed = 42
    classes = ('healthy', 'diseased')
    batch_size = 20
    n_classes = 2
    filename_roar_10 = 'data/trained_model_roar10.sav'
    filename_ori = 'data/trained_model_original.sav'
    train_labels, valid_labels, all_labels = load_labels()
    # save the explainer images of the figures
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    path_root = './data/exp/'
    subpath_heapmaps = 'heapmaps/heapmaps.pkl'
    explainers = ['Original', 'saliency', 'IntegratedGradients', 'NoiseTunnel', 'GuidedGradCam', 'GradCam',
                  'Noise Tunnel stev 2']
    image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']

    # loaded needed data
    print('loading training data')
    train_ds = Spectralloader(train_labels, root, mode)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    print('loading validation dataset')
    val_ds = Spectralloader(valid_labels, root, mode)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    print('loading complete dataset')
    all_ds = Spectralloader(all_labels, root, mode)
    all_dl = DataLoader(
        all_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # train model or use trained model from last execution
    if retrain:
        model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, "original")
        pickle.dump(model, open(filename_ori, 'wb'))
        print('trained model saved')
    else:
        model = pickle.load(open(filename_ori, 'rb'))

    # save the created explainer Image
    if plot_classes:
        # evaluate images and their classification
        print('creating explainer plots for specific classes')
        plot_explained_categories(model, val_dl, DEVICE, plot_classes, plot_classes, plot_classes, explainers)
    if plot_for_image_id:
        print('creating explainer plots for specified images')
        plot_explained_images(model, all_ds, DEVICE, explainers, image_ids)

    if roar_create_mask:
        create_mask(model, all_ds, path_root, subpath_heapmaps, DEVICE)
    if roar_train:
        with open(path_root + subpath_heapmaps, 'rb') as f:
            mask = pickle.load(f)
            print('applying ROAR to training DS')
            train_ds.apply_roar(10, mask)
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            print('applying ROAR to validation DS')
            val_ds.apply_roar(10, mask)
            val_dl = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, "10%removed")
            pickle.dump(model, open(filename_roar_10, 'wb'))
            print('trained model saved')


main()

# for img in imgs:
#     # you can show every image
#     img.show()

# Creating data indices for training and validation splits:
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                            sampler=train_sampler)
# validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                 sampler=valid_sampler)
# dataset_size = len(train_ds)
# indices = list(range(dataset_size))
# dl = DataLoader(train_ds, batch_size=10, shuffle=False, num_workers=4, drop_last=True)
