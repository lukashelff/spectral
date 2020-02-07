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
from cnn import *
from explainer import *
from plots import *
from helpfunctions import *

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
retrain = False
plot_for_image_id, plot_classes, plot_categories = False, False, False
roar_create_mask = False
roar_train = False
plot_roar_curve = False
roar_plot = True
N_EPOCHS = 120
lr = 0.00015



# roar_values = [10, 20]

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
    roar_explainers = ['noisetunnel', 'random', 'gradcam', 'guided_gradcam', 'noisetunnel_gaussian',
                       'guided_gradcam_gaussian']
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    # roar_explainers = ['noisetunnel', 'random', 'gradcam', 'guided_gradcam', 'noisetunnel_gaussian',
    #                    'guided_gradcam_gaussian']
    # roar_values = [10, 30, 50, 60, 80, 90]
    mode = 'rgb'
    shuffle_dataset = True
    random_seed = 42
    classes = ('healthy', 'diseased')
    batch_size = 20
    n_classes = 2
    trained_roar_models = './data/models/trained_model_roar'
    original_trained_model = './data/models/trained_model_original.sav'
    train_labels, valid_labels, all_labels = load_labels()
    # save the explainer images of the figures
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    path_exp = './data/exp/'
    subpath_heapmaps = 'heapmaps/heapmaps'
    explainers = ['Original', 'saliency', 'IntegratedGradients', 'NoiseTunnel', 'GuidedGradCam', 'GradCam',
                  'Noise Tunnel stev 2']
    # image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
    image_ids = ['Z17_1_0_0', 'Z15_2_1_2']

    if retrain or plot_classes or plot_categories:
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
        model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, "original")
        if not os.path.exists('./data/models/'):
            os.makedirs('./data/models/')
        pickle.dump(model, open(original_trained_model, 'wb'))
        print('trained model saved')
    if plot_categories or plot_classes or plot_for_image_id or roar_create_mask:
        model = pickle.load(open(original_trained_model, 'rb'))

    # save the created explainer Image
    if plot_classes or plot_categories:
        # evaluate images and their classification
        print('creating explainer plots for specific classes')
        plot_explained_categories(model, val_dl, DEVICE, plot_categories, plot_classes, plot_categories, explainers)
    if plot_for_image_id:
        print('creating explainer plots for specified images')
        plot_explained_images(model, all_ds, DEVICE, explainers, image_ids, 'original')

    if roar_create_mask:

        print('creating heap map for ROAR')
        create_mask(model, all_ds, path_exp, subpath_heapmaps, DEVICE, roar_explainers)
        print('heapmaps for ROAR created')

    if roar_train:
        for i in roar_explainers:
            train_roar_ds(path_exp + subpath_heapmaps + i + '.pkl', root, roar_values, trained_roar_models,
                          valid_labels, train_labels, batch_size, n_classes, N_EPOCHS, lr, mode, DEVICE, i)
    if plot_roar_curve:
        plot_dev_acc(roar_values, roar_explainers)

    if roar_plot:
        roar_explainers = ['noisetunnel', 'random', 'gradcam', 'guided_gradcam', 'noisetunnel_gaussian']
        roar_values = [10, 30, 50, 70, 90, 100]
        subpath = 'roar/'
        print('plotting modified images according to roar')
        w, h = 9 * len(roar_explainers), 8 * len(roar_values) + 8
        # fig = plt.figure(figsize=(w, h))
        # # fig.suptitle("image modification according to ROAR framework", fontsize=35)
        # if not os.path.exists(path_exp + subpath):
        #     os.makedirs(path_exp + subpath)
        # for c_ex, ex in enumerate(roar_explainers):
        #     # loading heapmap of corresponding explainer
        #     with open(path_exp + subpath_heapmaps + ex + '.pkl', 'rb') as f:
        #         mask = pickle.load(f)
        #         print('appling ' + ex + ' to image')
        #         for c_r, i in enumerate(roar_values):
        #             id = str(3) + '_' + image_ids[1]
        #             all_ds = Spectralloader(all_labels, root, mode)
        #             sub_path = str(i) + '%_of_' + ex + '.sav'
        #             path = './data/plots/values/' + sub_path
        #             if i == 0:
        #                 path = './data/plots/values/' + 'original.sav'
        #             else:
        #                 all_ds.apply_roar_single_image(i, mask, id)
        #             image, label = all_ds.get_by_id(id)
        #             acc = pickle.load(open(path, 'rb'))
        #             # create ROAR plot
        #             ax = fig.add_subplot(len(roar_values), len(roar_explainers),
        #                                  (c_ex + 1) + c_r * len(roar_explainers))
        #             ax.tick_params(axis='both', which='both', length=0)
        #             if c_ex == 0:
        #                 ax.set_ylabel(str(i) + '%')
        #             if c_r == 0:
        #                 ax.set_title(ex + '\n' + str(acc) + '%')
        #             else:
        #                 ax.set_title(str(acc) + '%')
        #             plt.imshow(np.transpose(image, (1, 2, 0)))
        #             plt.setp(ax.get_xticklabels(), visible=False)
        #             plt.setp(ax.get_yticklabels(), visible=False)
        # plt.show()
        # fig.savefig(path_exp + subpath + 'comparison_roar_images2' + '.png')
        # plt.close('all')
        roar_values = [0, 10, 30, 50, 70, 90, 100]

        fig2 = plt.figure(figsize=(w, h))
        # fig2.suptitle("modified image according to ROAR framework with applied interpretation of its saliency method",
        #               fontsize=35)
        for c_ex, ex in enumerate(roar_explainers):
            # loading heapmap of corresponding explainer
            with open(path_exp + subpath_heapmaps + ex + '.pkl', 'rb') as f:
                mask = pickle.load(f)
                print('applying ' + ex + ' to image')
                for c_r, i in enumerate(roar_values):
                    id = str(3) + '_' + image_ids[1]
                    all_ds = Spectralloader(all_labels, root, mode)
                    # loading model of explainer for corresponding remove value
                    if i == 0:
                        model = pickle.load(open(original_trained_model, 'rb'))
                    else:
                        model = pickle.load(open(trained_roar_models + '_' + ex + '_' + str(i) + '.sav', 'rb'))
                        all_ds.apply_roar_single_image(i, mask, id)
                    image, label = all_ds.get_by_id(id)
                    print('image with ' + str(i) + ' % of the image features removed')
                    # plot_explained_images(model, all_ds, DEVICE, explainers, image_ids, str(i) + "%removed")
                    model.to(DEVICE)
                    image = torch.from_numpy(image).to(DEVICE)
                    activation_map = explain_single(model, image, label, ex)
                    org = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))
                    org_img_edged = preprocessing.scale(np.array(org, dtype=float)[:, :, 1] / 255)
                    org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
                    # Compute the Canny filter for two values of sigma
                    org_img_edged = feature.canny(org_img_edged, sigma=3)
                    ax2 = fig2.add_subplot(len(roar_values), len(roar_explainers),
                                           (c_ex + 1) + c_r * len(roar_explainers))
                    ax2.tick_params(axis='both', which='both', length=0)
                    if c_ex == 0:
                        ax2.set_ylabel(str(i) + '%')
                    if c_r == 0:
                        ax2.set_title(ex)
                    ax2.imshow(org_img_edged, cmap=plt.cm.binary)
                    ax2.imshow(activation_map, cmap='viridis', vmin=np.min(activation_map), vmax=np.max(activation_map),
                               alpha=0.4)
                    plt.setp(ax2.get_xticklabels(), visible=False)
                    plt.setp(ax2.get_yticklabels(), visible=False)

        fig2.savefig(path_exp + subpath + 'comparison_explained_roar_images2' + '.png')
        plt.show()


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
