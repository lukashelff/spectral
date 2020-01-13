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
from explainer import explain
from plots import *

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
retrain = False
reexplain = True
plot_for_image_id, plot_classes, plot_healthy, plot_diseased = True, False, False, False


# cuda:1

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


# transpose image to display learned images
def display_rgb_grid(img, title):
    print(title + ' image displayed with shape ' + str(img.shape))
    plt.title(title)
    npimg = img.numpy()
    # plt.imshow(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# display rgb image
def display_rgb(img, title):
    plt.title(title)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


# display spectral image
def display_spec(img, transpose=True):
    import spectral
    im = img[:, :, [50, 88, 151]]
    # im = img[:, :, [24, 51, 118]]
    plt.imshow(im)
    plt.show()


def is_float(string):
    try:
        return float(string)
    except ValueError:
        return False


# create canvas man to show saved figures
def show_figure(fig, ax):
    # create a dummy figure and use its
    # manager to display the original figure
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    # new_manager.canvas.axes = ax
    fig.set_canvas(new_manager.canvas)


def main():
    mode = 'rgb'
    shuffle_dataset = True
    random_seed = 42
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    classes = ('healthy', 'diseased')
    batch_size = 20
    n_classes = 2
    N_EPOCHS = 200
    lr = 0.0001
    filename = 'data/trained_model.sav'
    train_labels, valid_labels, all_labels = load_labels()

    # loaded needed data
    if retrain:
        print('loading training data')
        # load train dataset
        train_ds = Spectralloader(train_labels, root, mode)
        # dataloader
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        print('loading validation dataset')
        # load valid dataset
        val_ds = Spectralloader(valid_labels, root, mode)
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
    if reexplain:
        print('loading complete dataset')
        # whole Dataset
        all_ds = Spectralloader(all_labels, root, mode)

        if plot_classes or plot_healthy or plot_diseased:
            val_ds = Spectralloader(valid_labels, root, mode)
            val_dl = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )

    # train model or use trained model from last execution
    if retrain:
        model = train(batch_size, n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE)
        pickle.dump(model, open(filename, 'wb'))
        print('trained model saved')
    else:
        model = pickle.load(open(filename, 'rb'))

    # save the explainer images of the figures
    path_root = './data/exp/'
    subpath_healthy = 'healthy/'
    subpath_diseased = 'diseased/'
    subpath_classification = 'classification/'
    subpath_single_image = 'single_image/'
    image_class = ['tp', 'fp', 'tn', 'fn']
    explainers = ['Original', 'saliency', 'IntegratedGradients', 'NoiseTunnel', 'GuidedGradCam', 'GradCam',
                  'GradCam Layer 4 Output']
    image_ids = ['Z18_4_1_1']
    # image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
    image_labels = np.zeros((len(image_ids), 4))
    image_pred = np.zeros((len(image_ids), 4))
    image_prob = np.zeros((len(image_ids), 4))
    number_images = 6  
    image_indexed = []
    for i in range(1, number_images + 1):
        image_indexed.append(str(i))

    # save the created explainer Image
    if reexplain:
        if plot_healthy or plot_diseased or plot_classes:
            # evaluate images and their classification
            print('creating explainer plots')
            evaluate(model, val_dl, number_images, explainers, image_class, path_root, subpath_healthy,
                     subpath_diseased, subpath_classification, DEVICE, plot_diseased, plot_healthy, plot_classes)
        if plot_for_image_id:
            print('creating explainer plots for specified images')
            # evaluate for specific Image IDs
            for i, id in enumerate(image_ids):
                for k in range(1, 5):
                    label, pred, prob = evaluate_id(str(k) + '_' + id, all_ds, model, explainers, path_root,
                                                    subpath_single_image + id + '/', DEVICE)
                    image_labels[i, k - 1] = label
                    image_pred[i, k - 1] = pred
                    image_prob[i, k - 1] = prob

    print('creating comparator of explainer plots')
    if plot_for_image_id:
        # plot created explainer
        for i, id in enumerate(image_ids):
            image_names = []
            for k in range(1, 5):
                image_names.append(str(k) + '_' + id)
            prediction = ''
            c1 = 'Truth for each day: '
            c2 = 'Prediction for each day: '
            prob = 'Probability of the prediction: '
            for k in range(0, 4):
                if image_pred[i, k] != -1:
                    c1 = c1 + 'Day ' + str(k) + ': ' + classes[int(image_labels[i, k])] + ' '
                    c2 = c2 + 'Day ' + str(k) + ': ' + classes[int(image_pred[i, k])] + ' '
                    prob = prob + 'Day ' + str(k) + ': ' + str(round(image_prob[i, k] * 100, 2)) + ' '
            prediction = c1 + '\n' + c2 + '\n' + prob
            plot_single_explainer(path_root, subpath_single_image + id + '/', explainers, image_names,
                                  'Plant comparison over days of ID: ' + id + '\n' + prediction)
            # 'Leaf is ', classes[predicted[ind]],
            #       'with a Probability of:', torch.max(F.softmax(outputs, 1)).item()
    if plot_classes:
        plot_single_explainer(path_root, subpath_classification, explainers, image_class,
                              'Class comparison TP, FP, TN, FN on plant diseases')
    if plot_diseased:
        plot_single_explainer(path_root, subpath_diseased, explainers, image_indexed,
                              'comparison between detected diseased images')
    if plot_healthy:
        plot_single_explainer(path_root, subpath_healthy, explainers, image_indexed,
                              'comparison between detected healthy images')


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
