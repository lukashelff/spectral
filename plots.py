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


def evaluate(model, val_dl, k, explainers, image_class, path_root, subpath_healthy, subpath_diseased,
             subpath_classification, DEVICE, plot_diseased, plot_healthy, plot_classes):
    # get index for each class
    # actual: healthy prediction: healthy, true positive
    # actual: diseased prediction: healthy, false positive
    # actual: diseased prediction: diseased, true negative
    # actual: healthy prediction: diseased, false negative
    index_classes = [-1, -1, -1, -1]
    index_classes_image = [0, 0, 0, 0]
    # 6 images of plants with detected diseased TP
    index_diseased = []
    index_diseased_image = []
    # 6 images of healthy plants TN
    index_healthy = []
    index_healthy_image = []
    # Predict val dataset with final trained resnet, 1 = disease, 0 = no disease
    model.to(DEVICE)
    model.eval()
    pred, labels = [], []
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_ = model(X)
            _, y_label_ = torch.max(y_, 1)
            pred += y_label_.tolist()
            labels += y.tolist()
            ydata, preddata = y.tolist(), y_label_.tolist()
            # get images and labels for images which get displayed
            for i in range(0, len(y)):
                if plot_diseased and len(index_diseased) < k and preddata[i] == 1 and ydata[i] == 1:
                    index_diseased += [ydata[i]]
                    index_diseased_image.append(explain(model, X[i], ydata[i]))
                if plot_healthy and len(index_healthy) < k and preddata[i] == 0 and y[i] == 0:
                    index_healthy += [ydata[i]]
                    index_healthy_image.append(explain(model, X[i], ydata[i]))
                if plot_classes and index_classes[2] == -1 and preddata[i] == 1 and ydata[i] == 1:
                    index_classes[2] = ydata[i]
                    index_classes_image[2] = explain(model, X[i], ydata[i])
                if plot_classes and index_classes[3] == -1 and preddata[i] == 0 and ydata[i] == 1:
                    index_classes[3] = ydata[i]
                    index_classes_image[3] = explain(model, X[i], ydata[i])
                if plot_classes and index_classes[0] == -1 and preddata[i] == 0 and ydata[i] == 0:
                    index_classes[0] = ydata[i]
                    index_classes_image[0] = explain(model, X[i], ydata[i])
                if plot_classes and index_classes[1] == -1 and preddata[i] == 1 and ydata[i] == 0:
                    index_classes[1] = ydata[i]
                    index_classes_image[1] = explain(model, X[i], ydata[i])

    if plot_healthy:
        # save images of explainer in data
        # save healthy images which got detected
        for k in range(0, len(index_healthy)):
            for i in range(0, len(explainers)):
                index_healthy_image[k][i].savefig(
                    path_root + subpath_healthy + explainers[i] + str(k) + '.png',
                    bbox_inches='tight')

    if plot_diseased:
        # save images of explainer in data
        # save diseased images which got detected
        for k in range(0, len(index_diseased)):
            for i in range(0, len(explainers)):
                index_diseased_image[k][i].savefig(
                    path_root + subpath_diseased + explainers[i] + str(k) + '.png',
                    bbox_inches='tight')

    if plot_classes:
        # save images of every class to compare
        for k in range(0, len(image_class)):
            for i in range(0, len(explainers)):
                index_classes_image[k][i].savefig(
                    path_root + subpath_classification + explainers[i] + image_class[k] + '.png', bbox_inches='tight')


def evaluate_id(image_id, ds, model, explainers, path_root, subpath, DEVICE):
    # evaluate predictions with model and create Images from explainers
    # # get first batch for evaluation
    # dataiter = iter(val_dl)
    # image1, label1 = next(dataiter)
    #
    # # print images of first batch
    # display_rgb_grid(torchvision.utils.make_grid(image1), 'images of batch 1')
    # # predict images of first batch
    # print('GroundTruth: ', ' '.join('%5s' % classes[label1[j]] for j in range(batch_size)))
    #
    # outputs1 = model(image1.to(DEVICE))
    #
    # _, predicted1 = torch.max(outputs1, 1)
    #
    # print('Predicted: ', ' '.join('%5s' % classes[predicted1[j]] for j in range(batch_size)))

    if not os.path.exists(path_root + subpath):
        os.makedirs(path_root + subpath)
    image, label = ds.get_by_id(image_id)
    if image is not None:
        model.to(DEVICE)
        image = torch.from_numpy(image).to(DEVICE)
        explained = explain(model, image, label)
        for i in range(0, len(explainers)):
            directory = path_root + subpath + explainers[i] + image_id + '.png'
            explained[i].savefig(directory, bbox_inches='tight')
        image = image[None]
        image = image.type('torch.FloatTensor').to(DEVICE)
        # print(model.get_device())
        output = model(image)
        _, pred = torch.max(output, 1)
        prob = torch.max(F.softmax(output, 1)).item()
        return label, pred.item(), prob
    else:
        return -1, -1, -1


# compare given images of plantes in a plot
def plot_single_explainer(pathroot, subpath, explainers, image_names, title):
    exnum = len(explainers)
    number_images = len(image_names)
    images = []
    for k in range(0, number_images):
        for i in range(0, exnum):
            try:
                images.append(mpimg.imread(pathroot + subpath + explainers[i] + image_names[k] + '.png'))
            except FileNotFoundError:
                print('image could not be loaded')
    number_images = len(images) // exnum
    fig = plt.figure(figsize=(6 * exnum + 2, 7 * number_images + 8))
    fig.suptitle(title, fontsize=40)
    for k in range(0, len(images) // exnum):
        for i in range(0, exnum):
            ax = fig.add_subplot(number_images, exnum, (i + 1) + k * exnum)
            plt.imshow(images[i + k * exnum])
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if k == 0:
                ax.set_title(explainers[i], fontsize=25)
            if i == 0:
                if subpath == 'classification/':
                    if image_names[k] == 'tp':
                        ax.set_ylabel('TP:\n Truth: healthy\n Prediction: healthy', fontsize=25)
                    if image_names[k] == 'fp':
                        ax.set_ylabel('FP:\n Truth: healthy\n Prediction: diseased', fontsize=25)
                    if image_names[k] == 'tn':
                        ax.set_ylabel('TN:\n Truth: diseased\n Prediction: diseased', fontsize=25)
                    if image_names[k] == 'fn':
                        ax.set_ylabel('FN:\n Truth: diseased\n Prediction: healthy', fontsize=25)
                else:
                    ax.set_ylabel('image ' + image_names[k], fontsize=25)

    fig.tight_layout()
    fig.savefig(pathroot + subpath + 'conclusion' + '.png')
    plt.show()
