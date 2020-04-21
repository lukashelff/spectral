import os
import pickle
from os import listdir
from typing import List, Any

import torchvision
from PIL import Image as PImage
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as t_datasets
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
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import torchvision.datasets as t_datasets
import torch.utils.data as data
from roar import *
from spectralloader import *
from cnn import *
from explainer import *
from plots import *
from helpfunctions import *


def main():
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    modes = ['plants', 'imagenet']
    mode = modes[1]
    # resizes all images and replaces them in folder
    resize_imagenet = False
    retrain = True
    plot_for_image_id, plot_classes, plot_categories = False, False, False
    roar_create_mask = False
    roar_train = False
    plot_roar_curve = False
    roar_mod_im_comp = False
    roar_expl_im = False
    # CNN default learning parameters
    N_EPOCHS = 120
    lr = 0.00015
    n_classes = 2
    batch_size = 10
    cv_iterations_total = 5
    test_size = 482
    classes = ('healthy', 'diseased')

    # Training Values for plant dataset, resnet18 with lr = 0.00015, Epochs = 120, batchsize = 20
    roar_explainers = ['gradcam', 'guided_gradcam', 'guided_gradcam_gaussian',
                       'noisetunnel', 'noisetunnel_gaussian', 'Integrated_Gradients']
    roar_explainers = ['gradcam', 'guided_gradcam', 'guided_gradcam_gaussian',
                       'noisetunnel', 'random', 'Integrated_Gradients']
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    roar_values = [10, 30, 70, 90]
    cv_it_to_calc = [0]
    if mode == 'imagenet':
        if resize_imagenet:
            val_format()
            upscale_imagenet()
        n_classes = 200
        N_EPOCHS = 20
        lr = 0.001
        batch_size = 20
        # print('nr epochs: ' + str(N_EPOCHS))
        # print('batch_size ' + str(batch_size))
        # print('lr ' + str(lr))
        cv_iterations_total = 1
        test_size = 10000
        # train_imagenet(N_EPOCHS, lr, batch_size, DEVICE, mode)

    train_labels, valid_labels, all_data, labels = load_labels(mode)
    sss = StratifiedShuffleSplit(n_splits=cv_iterations_total, test_size=test_size, random_state=0)
    # save the explainer images of the figures
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    if not os.path.exists('./data/' + mode + '/' + 'models/'):
        os.makedirs('./data/' + mode + '/' + 'models/')
    if not os.path.exists('./data/' + mode + '/' + 'plots/'):
        os.makedirs('./data/' + mode + '/' + 'plots/')
    if not os.path.exists('./data/' + mode + '/' + 'plots/values/'):
        os.makedirs('./data/' + mode + '/' + 'plots/values/')

    trained_roar_models = './data/' + mode + '/' + 'models/trained_model_roar'
    original_trained_model = './data/' + mode + '/' + 'models/trained_model_original.pt'
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    path_exp = './data/' + mode + '/' + 'exp/'
    subpath_heatmaps = 'heatmaps/heatmaps'
    explainers = ['Original', 'saliency', 'IntegratedGradients', 'NoiseTunnel', 'GuidedGradCam', 'GradCam',
                  'Noise Tunnel stev 2']
    image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']

    # loading Datasets
    if plot_classes or plot_categories:
        # load needed data
        print('loading validation dataset')
        val_ds = Spectralloader(valid_labels, root, mode)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
    if plot_for_image_id or roar_create_mask:
        print('loading whole dataset')
        all_ds = Spectralloader(all_data, root, mode)

    # train model or use trained model from last execution
    if retrain:
        train_cross_val(sss, all_data, labels, root, mode, batch_size, n_classes, N_EPOCHS, lr, DEVICE,
                        original_trained_model, cv_it_to_calc)

    if plot_categories or plot_classes or plot_for_image_id or roar_create_mask:
        original_model = get_model(DEVICE, n_classes, mode)
        original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))

    # save the created explainer Image
    if plot_classes or plot_categories:
        # evaluate images and their classification
        print('creating explainer plots for specific classes')
        plot_explained_categories(original_model, val_dl, DEVICE, plot_categories, plot_classes, plot_categories,
                                  explainers, mode)
    if plot_for_image_id:
        print('creating explainer plots for specified images')
        plot_explained_images(original_model, all_ds, DEVICE, explainers, image_ids, 'original', mode)

    # create a mask containing the heatmap of all specified images
    if roar_create_mask:
        print('creating for ROAR mask')
        create_mask(original_model, all_ds, path_exp, subpath_heatmaps, DEVICE, roar_explainers)
        print('mask for ROAR created')

    # ROAR remove and retrain applied to all specified explainers and remove percentages
    if roar_train:
        train_roar_ds(path_exp + subpath_heatmaps, roar_values, trained_roar_models, all_data, labels, batch_size,
                      n_classes, N_EPOCHS, lr, DEVICE, roar_explainers, sss, root, mode, cv_it_to_calc)

    # plot the acc curves of all trained ROAR models
    if plot_roar_curve:
        plot_dev_acc(roar_values, roar_explainers, cv_iterations_total, mode)

    # comparison of modified roar Images
    if roar_mod_im_comp:
        print('creating ROAR comparison plot')
        roar_comparison(mode, roar_explainers, cv_iterations_total)

    # interpretation/explaination of modified roar Images
    if roar_expl_im:
        print('creating ROAR explanation plot')
        roar_comparison_explained(mode, DEVICE, roar_explainers)


if __name__ == '__main__':
    main()
