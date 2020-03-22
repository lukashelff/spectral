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
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import feature
import multiprocessing as mp
import torch

from roar import *
from spectralloader import Spectralloader
from cnn import *
from explainer import *
from plots import *
from helpfunctions import *

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
retrain = False
plot_for_image_id, plot_classes, plot_categories = False, False, False
roar_create_mask = True
roar_train = True
plot_roar_curve = True
roar_mod_im_comp = False
roar_expl_im = False
N_EPOCHS = 120
lr = 0.00015



# cuda:1
#
# returns Array of tuples(String, int) with ID and disease information 0 disease/ 1 healthy e.g. (3_Z2_1_0_1, 0)
def load_labels():
    mp.set_start_method('spawn')
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
    roar_explainers = ['gradcam', 'guided_gradcam', 'guided_gradcam_gaussian',
                       'noisetunnel', 'noisetunnel_gaussian', 'Integrated_Gradients']
    roar_explainers = ['LRP']
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    mode = 'rgb'
    shuffle_dataset = True
    random_seed = 42
    classes = ('healthy', 'diseased')
    batch_size = 20
    n_classes = 2
    trained_roar_models = './data/models/trained_model_roar'
    original_trained_model = './data/models/trained_model_original.pt'
    train_labels, valid_labels, all_labels = load_labels()
    # save the explainer images of the figures
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    path_exp = './data/exp/'
    subpath_heapmaps = 'heapmaps/heapmaps'
    explainers = ['Original', 'saliency', 'IntegratedGradients', 'NoiseTunnel', 'GuidedGradCam', 'GradCam',
                  'Noise Tunnel stev 2']
    image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']

    # loading Datasets
    if retrain or plot_classes or plot_categories or roar_train:
        # loaded needed data
        print('loading training data')
        train_ds = Spectralloader(train_labels, root, mode)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
        print('loading validation dataset')
        val_ds = Spectralloader(valid_labels, root, mode)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
    if plot_for_image_id or roar_create_mask:
        print('loading whole dataset')
        all_ds = Spectralloader(all_labels, root, mode)

    # train model or use trained model from last execution
    if retrain:
        # train on original data
        original_model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, "original")
        if not os.path.exists('./data/models/'):
            os.makedirs('./data/models/')
        torch.save(original_model.state_dict(), original_trained_model)
        print('trained model saved')
    if plot_categories or plot_classes or plot_for_image_id or roar_create_mask:
        original_model = models.resnet18(pretrained=True)
        original_model.fc = nn.Linear(512, n_classes)
        original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))

    # save the created explainer Image
    if plot_classes or plot_categories:
        # evaluate images and their classification
        print('creating explainer plots for specific classes')
        plot_explained_categories(original_model, val_dl, DEVICE, plot_categories, plot_classes, plot_categories,
                                  explainers)
    if plot_for_image_id:
        print('creating explainer plots for specified images')
        plot_explained_images(original_model, all_ds, DEVICE, explainers, image_ids, 'original')

    # create a mask containing the heapmap of all specified images
    if roar_create_mask:
        print('creating for ROAR mask')
        create_mask(original_model, all_ds, path_exp, subpath_heapmaps, DEVICE, roar_explainers)
        print('mask for ROAR created')

    # ROAR remove and retrain applied to all specified explainers and remove percentages
    if roar_train:
        train_roar_ds(path_exp + subpath_heapmaps, roar_values, trained_roar_models, val_ds, train_ds, batch_size,
                      n_classes, N_EPOCHS, lr, DEVICE, roar_explainers)

    # plot the acc curves of all trained ROAR models
    if plot_roar_curve:
        plot_dev_acc(roar_values, roar_explainers)

    # comparison of modified roar Images
    if roar_mod_im_comp:
        print('creating ROAR comparison plot')
        eval_roar_mod_im_comp(mode, roar_explainers)

    # interpretation/explaination of modified roar Images
    if roar_expl_im:
        print('creating ROAR explanation plot')
        eval_roar_expl_im(mode, DEVICE, roar_explainers)


if __name__ == '__main__':
    main()
