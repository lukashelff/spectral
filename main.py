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


def evaluate(model, val_dl, k, explainers, image_class, path_root, subpath_healthy, subpath_diseased,
             subpath_classification):
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


def evaluate_id(image_id, ds, model, explainers, path_root, subpath):
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


def main():
    mode = 'rgb'
    shuffle_dataset = True
    random_seed = 42
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    classes = ('healthy', 'diseased')
    batch_size = 30
    n_classes = 2
    N_EPOCHS = 500
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
        # Dataset with all data
        all_ds = Spectralloader(all_labels, root, mode)
        all_dl = DataLoader(
            all_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
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
                  'Noise Tunnel stev 2']
    # image_ids = ['Z18_4_1_1']
    image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
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
                     subpath_diseased,
                     subpath_classification)
        if plot_for_image_id:
            print('creating explainer plots for specified images')
            # evaluate for specific Image IDs
            for i, id in enumerate(image_ids):
                for k in range(1, 5):
                    label, pred, prob = evaluate_id(str(k) + '_' + id, all_ds, model, explainers, path_root,
                                                    subpath_single_image + id + '/')
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
