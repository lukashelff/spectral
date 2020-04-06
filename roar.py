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
from cnn import *
from explainer import *
from plots import *
from helpfunctions import *

# roar_explainers = ['guided_gradcam', 'random', 'gradcam', 'noisetunnel_gaussian',
#                    'guided_gradcam_gaussian', 'noisetunnel', 'Integrated_Gradients']
image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
image_ids_roar_exp = [0, 3, 4, 6]
image_labels = [('3_Z18_4_1_1', 1), ('3_Z17_1_0_0', 1), ('3_Z16_2_1_1', 1), ('3_Z15_2_1_2', 1), ('3_Z8_4_0_0', 1),
                ('3_Z8_4_1_2', 1), ('3_Z1_3_1_1', 0), ('3_Z2_1_0_2', 0)]
trained_roar_models = './data/models/trained_model_roar'
original_trained_model = './data/models/trained_model_original.pt'
root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
path_exp = './data/exp/'
subpath_heapmaps = 'heapmaps/heapmaps'
subpath = 'roar/'
n_classes = 2


# applying the explainers to an roar trained image
# interpretation/explaination of modified roar Images
# Axes: removed % of image features and explainers
def roar_comparison_explained(mode, DEVICE, explainers):
    # explainers = ['noisetunnel', 'gradcam', 'guided_gradcam', 'noisetunnel_gaussian', 'guided_gradcam_gaussian']
    roar_expl_im_values = [0, 10, 20, 30, 50, 70, 90, 100]
    w, h = 8 * len(explainers), 7 * len(roar_expl_im_values) + 10
    for k in image_ids_roar_exp:
        id = str(3) + '_' + image_ids[k]
        fig = plt.figure(figsize=(w, h))
        fig.subplots_adjust(top=0.95)
        fig.suptitle(
            "modified image " + id + " according to ROAR framework with applied interpretation of its saliency method",
            fontsize=80)
        print('plotting modified image:' + id + ' according to roar')
        all_ds = Spectralloader([image_labels[k]], root, mode)
        image, label = all_ds.get_by_id(id)
        for c_e, a in enumerate(explainers):
            ax = fig.add_subplot(len(roar_expl_im_values) + 1, len(explainers),
                                 c_e + 1)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(a, fontsize=40)
            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if c_e == 0:
                ax.set_ylabel('original image', fontsize=40)
        for c_ex, ex in enumerate(explainers):
            # loading heapmap of corresponding explainer
            with open(path_exp + subpath_heapmaps + ex + '.pkl', 'rb') as f:
                mask = pickle.load(f)
                print('applying ' + ex + ' to image')
                for c_r, i in enumerate(roar_expl_im_values):
                    # select 3 day image of image ID
                    # loading model of explainer for corresponding remove value
                    all_ds = Spectralloader([image_labels[k]], root, mode)
                    if i == 0:
                        model = get_model(DEVICE, n_classes)
                        model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
                    else:
                        model = get_model(DEVICE, n_classes)
                        model.load_state_dict(
                            torch.load(trained_roar_models + '_' + ex + '_' + str(i) + '.pt', map_location=DEVICE))
                        all_ds.apply_roar_single_image(i, mask, id, 'mean', ex)
                    # plot_explained_images(model, all_ds, DEVICE, explainers, image_ids, str(i) + "%removed")
                    image, label = all_ds.get_by_id(id)
                    model.to(DEVICE)
                    image = torch.from_numpy(image).to(DEVICE)
                    activation_map = explain_single(model, image, label, ex, True)
                    org = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))
                    org_img_edged = preprocessing.scale(np.array(org, dtype=float)[:, :, 1])
                    org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
                    # Compute the Canny filter for two values of sigma
                    org_img_edged = feature.canny(org_img_edged, sigma=3)
                    ax = fig.add_subplot(len(roar_expl_im_values) + 1, len(explainers),
                                         (c_ex + 1) + (c_r + 1) * len(explainers))
                    ax.tick_params(axis='both', which='both', length=0)
                    if c_ex == 0:
                        ax.set_ylabel(str(i) + '%', fontsize=40)
                    ax.imshow(org_img_edged, cmap=plt.cm.binary)
                    ax.imshow(activation_map, cmap='viridis', alpha=0.4)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)

        fig.savefig(path_exp + subpath + 'comparison_explained_roar_image_' + id + '.png')
        fig.clear()


# plotting the roar trained images
# comparison of modified roar Images
# Axes: removed % of image features and explainers
def roar_comparison(mode, roar_explainers, cv_iter):
    # roar_explainers = ['random'] + roar_explainers
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    print('plotting modified images according to roar')
    w, h = 8 * len(roar_explainers), 7 * len(roar_values) + 3
    for k in image_ids_roar_exp:
        fig = plt.figure(figsize=(w, h))
        fig.subplots_adjust(top=0.95)
        fig.suptitle("image " + image_ids[k] + " modificed according to ROAR framework", fontsize=80)
        print('modifing image: ' + image_ids[k])
        if not os.path.exists(path_exp + subpath):
            os.makedirs(path_exp + subpath)
        for c_ex, ex in enumerate(roar_explainers):
            # loading heapmap of corresponding explainer
            with open(path_exp + subpath_heapmaps + ex + '.pkl', 'rb') as f:
                mask = pickle.load(f)
                print('appling ' + ex + ' to image')
                for c_r, roar_per in enumerate(roar_values):
                    id = str(3) + '_' + image_ids[k]
                    all_ds = Spectralloader([image_labels[k]], root, mode)
                    sub_path = str(roar_per) + '%_of_' + ex + '.sav'
                    path = './data/plots/values/' + sub_path
                    if roar_per == 0:
                        acc = get_cross_val_acc('original', roar_per, cv_iter)
                    else:
                        all_ds.apply_roar_single_image(roar_per, mask, id, 'comp', ex)
                        acc = get_cross_val_acc(ex, roar_per, cv_iter)
                    image, label = all_ds.get_by_id(id)
                    # create ROAR plot
                    ax = fig.add_subplot(len(roar_values), len(roar_explainers),
                                         (c_ex + 1) + c_r * len(roar_explainers))
                    ax.tick_params(axis='both', which='both', length=0)
                    if c_ex == 0:
                        ax.set_ylabel(str(roar_per) + '%', fontsize=40)
                    if c_r == 0:
                        ax.set_title(ex + '\n' + str(acc) + '%', fontsize=40)
                    else:
                        ax.set_title(str(acc) + '%', fontsize=40)
                    plt.imshow(np.transpose(image, (1, 2, 0)))
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
        fig.savefig(path_exp + subpath + 'comparison_roar_images' + id + '.png')
        fig.clear()
