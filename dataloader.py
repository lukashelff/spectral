import pickle
from os import listdir
from typing import List, Any

import torchvision
from PIL import Image as PImage
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
from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift
from captum.attr import Saliency
from captum.attr import visualization as viz
from captum.attr import GuidedGradCam
from captum.attr._core.guided_grad_cam import LayerGradCam

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


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


def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)


def get_frozen(model_params):
    return (p for p in model_params if not p.requires_grad)


def all_trainable(model_params):
    return all(p.requires_grad for p in model_params)


def all_frozen(model_params):
    return all(not p.requires_grad for p in model_params)


def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False


# trains and returns model for the given dataloader and computes graph acc, balanced acc and loss
def train(batch_size, n_classes, N_EPOCHS, learning_rate, train_dl, val_dl):
    train_loss = np.zeros(N_EPOCHS)
    train_acc = np.zeros(N_EPOCHS)
    train_balanced_acc = np.zeros(N_EPOCHS)
    valid_loss = np.zeros(N_EPOCHS)
    valid_acc = np.zeros(N_EPOCHS)
    valid_balanced_acc = np.zeros(N_EPOCHS)

    def get_model():
        model = models.resnet18(pretrained=True)
        freeze_all(model.parameters())
        model.fc = nn.Linear(512, n_classes)
        model = model.to(DEVICE)
        return model

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        get_trainable(model.parameters()),
        lr=learning_rate,
        # momentum=0.9,
    )

    for epoch in range(N_EPOCHS):

        # Train
        model.train()

        total_loss, n_correct, n_samples, pred, all_y = 0.0, 0, 0, [], []
        for batch_i, (X, y) in enumerate(train_dl):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_ = model(X)
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()

            # Statistics
            # print(
            #     f"Epoch {epoch+1}/{N_EPOCHS} |"
            #     f"  batch: {batch_i} |"
            #     f"  batch loss:   {loss.item():0.3f}"
            # )
            _, y_label_ = torch.max(y_, 1)
            n_correct += (y_label_ == y).sum().item()
            total_loss += loss.item() * X.shape[0]
            n_samples += X.shape[0]
            pred += y_label_.tolist()
            all_y += y.tolist()

        train_balanced_acc[epoch] = balanced_accuracy_score(all_y, pred) * 100
        train_loss[epoch] = total_loss / n_samples
        train_acc[epoch] = n_correct / n_samples * 100

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} |"
            f"  train loss: {train_loss[epoch]:9.3f} |"
            f"  train acc:  {train_acc[epoch]:9.3f}% |"
            f"  balanced acc:  {train_balanced_acc[epoch]:9.3f}%"

        )

        # Eval
        model.eval()

        total_loss, n_correct, n_samples, pred, all_y = 0.0, 0, 0, [], []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                y_ = model(X)

                # Statistics
                _, y_label_ = torch.max(y_, 1)
                n_correct += (y_label_ == y).sum().item()
                loss = criterion(y_, y)
                total_loss += loss.item() * X.shape[0]
                n_samples += X.shape[0]
                pred += y_label_.tolist()
                all_y += y.tolist()

        valid_balanced_acc[epoch] = balanced_accuracy_score(all_y, pred) * 100
        valid_loss[epoch] = total_loss / n_samples
        valid_acc[epoch] = n_correct / n_samples * 100

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} |"
            f"  valid loss: {valid_loss[epoch]:9.3f} |"
            f"  valid acc:  {valid_acc[epoch]:9.3f}% |"
            f"  balanced acc:  {valid_balanced_acc[epoch]:9.3f}%"
        )

    # plot acc, balanced acc and loss
    plt.plot(train_acc, color='skyblue', label='train acc')
    plt.plot(valid_acc, color='orange', label='valid_acc')
    plt.plot(train_balanced_acc, color='darkblue', label='train_balanced_acc')
    plt.plot(valid_balanced_acc, color='red', label='valid_balanced_acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('./data/plots/accuracy.png')
    plt.show()
    plt.plot(train_loss, color='red', label='train_loss')
    plt.plot(valid_loss, color='orange', label='valid_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('./data/plots/loss.png')
    plt.show()

    return model


def explain(model, image, label):
    input = image.unsqueeze(0)
    input.requires_grad = True
    model.eval()

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  target=label,
                                                  **kwargs
                                                  )

        return tensor_attributions

    # saliency
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # IntegratedGradients
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # IntegratedGradients Noise Tunnel
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          # n_samples=100,
                                          stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # GuidedGradCam
    gc = GuidedGradCam(model, model.layer4)
    attr_gc = attribute_image_features(gc, input)
    attr_gc = np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # GradCam
    gco = LayerGradCam(model, model.layer4)
    attr_gco = attribute_image_features(gco, input)
    attr_gco = np.transpose(attr_gco.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # # Deeplift
    # dl = DeepLift(model)
    # attr_dl = attribute_image_features(dl, input)
    # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # print('Leaf is ', classes[predicted[ind]],
    #       'with a Probability of:', torch.max(F.softmax(outputs, 1)).item())

    original_image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
    # Original Image
    f1, a1 = viz.visualize_image_attr(None, original_image,
                                      method="original_image",
                                      # title="Original Image",
                                      use_pyplot=False)
    # Overlayed Gradient Magnitudes saliency
    f2, a2 = viz.visualize_image_attr(grads, original_image, sign="absolute_value", method="blended_heat_map",
                                      # show_colorbar=True, title="Overlayed Gradient Magnitudes saliency",
                                      use_pyplot=False)
    # Overlayed Integrated Gradients
    f3, a3 = viz.visualize_image_attr(attr_ig, original_image, sign="all", method="blended_heat_map",
                                      # show_colorbar=True, title="Overlayed Integrated Gradients",
                                      use_pyplot=False)
    # Overlayed Noise Tunnel
    f4, a4 = viz.visualize_image_attr(attr_ig_nt, original_image, sign="absolute_value",
                                      outlier_perc=10, method="blended_heat_map", use_pyplot=False,
                                      # title="Overlayed Noise Tunnel \n with SmoothGrad Squared", show_colorbar=True
                                      )
    # # DeepLift
    # f5, a5 = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all",
    #                                   # show_colorbar=True, title="Overlayed DeepLift"
    #                                   )

    # GuidedGradCam
    f6, a6 = viz.visualize_image_attr(attr_gc, original_image, sign="absolute_value", method="blended_heat_map",
                                      show_colorbar=False, use_pyplot=False)

    # GradCam
    f7, a7 = viz.visualize_image_attr(attr_gco, original_image, sign="absolute_value", method="blended_heat_map",
                                      show_colorbar=False, use_pyplot=False)

    return [f1, f2, f3, f4, f6, f7]


def plotexplainer(pathroot, explainers, imageclass, numdis):
    # compare images between classes
    classnum = len(imageclass)
    exnum = len(explainers)
    print('Ä')
    print(classnum)
    print(exnum)
    fig = plt.figure(figsize=(6 * exnum, 6 * classnum))
    # fig.set_title('Class comparison TP, FP, TN, FN plant images')
    print("hallo")
    for k in range(0, classnum):
        for i in range(0, exnum):
            img = mpimg.imread(pathroot + explainers[i] + imageclass[k] + '.png')
            ax = fig.add_subplot(classnum, exnum, (i + 1) + k * exnum)
            plt.imshow(img)
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if k == 0:
                ax.set_title(explainers[i], fontsize=25)
            if i == 0:
                if imageclass[k] == 'tp':
                    ax.set_ylabel('TP:\n Truth: healthy\n Prediction: healthy', fontsize=25)
                if imageclass[k] == 'fp':
                    ax.set_ylabel('FP:\n Truth: diseased\n Prediction: healthy', fontsize=25)
                if imageclass[k] == 'tn':
                    ax.set_ylabel('TN:\n Truth: diseased\n Prediction: diseased', fontsize=25)
                if imageclass[k] == 'fn':
                    ax.set_ylabel('FN:\n Truth: healthy\n Prediction: diseased', fontsize=25)
    fig.tight_layout()
    fig.savefig(pathroot + 'conclusion' + '.png')
    plt.show()

    # compare detected diseased images of plantes
    fig = plt.figure(figsize=(6 * exnum, 6 * numdis))
    # fig.set_title('comparison between detected diseases')
    for k in range(0, numdis):
        for i in range(0, exnum):
            img = mpimg.imread(pathroot + 'diseasedims/' + explainers[i] + 'diseased' + str(k) + '.png')
            ax = fig.add_subplot(numdis, exnum, (i + 1) + k * exnum)
            plt.imshow(img)
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if k == 0:
                ax.set_title(explainers[i], fontsize=25)
            if i == 0:
                ax.set_ylabel('diseased image ' + str(k), fontsize=25)
    fig.tight_layout()
    fig.savefig(pathroot + 'diseasedims/' + 'conclusion' + '.png')
    plt.show()

    # compare detected diseased images of plantes
    fig = plt.figure(figsize=(6 * exnum, 6 * numdis))
    # fig.set_title('comparison between not detected diseases')
    for k in range(0, numdis):
        for i in range(0, exnum):
            img = mpimg.imread(pathroot + 'diseasedimsnotdetected/' + explainers[i] + 'diseased' + str(k) + '.png')
            ax = fig.add_subplot(numdis, exnum, (i + 1) + k * exnum)
            plt.imshow(img)
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if k == 0:
                ax.set_title(explainers[i], fontsize=25)
            if i == 0:
                ax.set_ylabel('diseased image ' + str(k), fontsize=25)
    fig.tight_layout()
    fig.savefig(pathroot + 'notdetecteddiseasedims/' + 'conclusion' + '.png')
    plt.show()


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


class Spectralloader(Dataset):
    """
        The spektral dataset can be found in folder

        /home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/

        The folder structure of the dataset is as follows::

            └── VNIR
                ├── 1_Z1
                │   └── segmented_leafs
                │       ├── data.p
                │       └── memmap.dat
                ├── 1_Z2
                │   └── segmented_leafs
                │       ├── data.p
                │       └── memmap.dat
                ├──
                 ...
                ├──
                ├── 1_Z13
                │   └── segmented_leafs
                │       ├── data.p
                │       └── memmap.dat
                ├── 2_Z1
                │   └── segmented_leafs
                │       ├── data.p
                │       └── memmap.dat
                ├──
                ├──
                ├── 4_Z18
                │   └── segmented_leafs
                │       ├── data.p
                │       └── memmap.dat


                Platte Z1 - Z18
                Tag der Aufnahme: 1
                1_Z1 - 1_Z13
                Tag der Aufnahme: 2
                2_Z1 - 2_Z18
                Tag der Aufnahme: 3
                3_Z1 - 3_Z18
                Tag der Aufnahme: 4
                4_Z1 - 4_Z18


                train and valid Dataset

                Beispiel für ein Blatt:
                1_Z3_2_0_1;0
                    2_0_1 id des Blattes
                    0 steht für gesund; 1 für krank

                Image format
                (213, 255, 328)
                RGB format
                (213, 255, 3)
                learning format
                (3, 255, 213)
                RGB channels
                [50, 88, 151]
                SWIR
                [24, 51, 118]

        """

    def __init__(self, labels, root, mode, transform=None):
        # load data for given labels
        self.labels_ids, self.labels, self.data_ids, self.data = self.load_images_for_labels(root, labels, mode=mode)
        self.transform = transform
        self.path = root

    def __getitem__(self, index):
        # return only 1 sample and label (according to "Index")
        # get label for ID
        label = self.labels[index]
        ID = self.labels_ids[index]
        # get Image Index for Id of Label
        index_im = [i for i, s in enumerate(self.data_ids) if ID[0] == s]
        image = self.data[index_im[0]]

        return image, label

    def __len__(self):
        return len(self.labels)

        # image loader

    # returns 4 Arrays labelIdS, labels, ImageIDs and the Image as a Tuple(String, mmemap) e.g. (3_Z2_1_0_1, memmap)
    def load_images_for_labels(self, root_path, labels, mode):
        # loads all the images have existing entry labels
        def load_image(path):
            ids, ims = [], []
            dict = pickle.load(open(path + '/data.p', 'rb'))
            shape = dict['memmap_shape']
            samples = dict['samples']
            data_all = np.memmap(path + '/memmap.dat', mode='r', shape=shape, dtype='float32')
            labels_ids = [i[0] for i in labels]
            for k, i in enumerate(samples):
                # only add if we have a label for the image
                if i['id'].replace(',', '_') in labels_ids:
                    if mode == 'rgb':
                        # ims.append(data_all[k][:, :, [50, 88, 151]].reshape(3, 255, 213))
                        # ims.append(data_all[k][:, :, [50, 88, 151]])
                        data = np.transpose(data_all[k][:, :, [50, 88, 151]], (2, 0, 1))
                        ims.append(data)
                    elif mode == 'spec':
                        # ims.append(data_all[k].reshape(3, 255, 213))
                        ims.append(data_all[k])
                    ids.append(i['id'].replace(',', '_'))
            return ids, ims

        # removes label entries with no existing image
        def sync_labels(im_ids):
            labs = labels
            for (k, i) in labels:
                if k not in im_ids:
                    labs.remove((k, i))
            lab_raw = [i[1] for i in labs]
            return labs, lab_raw

        loaded_images = []
        loaded_image_ids = []
        for i in range(1, 5):
            print("loading images of day: " + str(i))
            if i == 1:

                for k in range(1, 14):
                    ids, im = load_image(root_path + str(i) + '_Z' + str(k) + '/segmented_leafs')
                    loaded_images += im
                    loaded_image_ids += ids
            else:
                for k in range(1, 19):
                    if not (k == 16 and i == 4):
                        ids, im = load_image(root_path + str(i) + '_Z' + str(k) + '/segmented_leafs')
                        loaded_images += im
                        loaded_image_ids += ids
        label_ids, label_raw = sync_labels(loaded_image_ids)
        print("all images loaded")
        return label_ids, label_raw, loaded_image_ids, loaded_images


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

    print('loading training data')
    # load train dataset
    train_labels, valid_labels, all_labels = load_labels()
    train_ds = Spectralloader(train_labels, root, mode)

    print('loading validation data')
    # load valid dataset
    val_ds = Spectralloader(valid_labels, root, mode)

    batch_size = 20
    n_classes = 2
    N_EPOCHS = 50
    lr = 0.00025
    retrain = False
    reexplain = False
    filename = 'data/trained_model.sav'

    # dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # train model or use trained model from last execution
    if retrain:
        model = train(batch_size, n_classes, N_EPOCHS, lr, train_dl, val_dl)
        pickle.dump(model, open(filename, 'wb'))
    else:
        model = pickle.load(open(filename, 'rb'))

    # get index for each class
    # actual: healthy prediction: healthy, true positive, image 8 of sixth batch
    # actual: diseased prediction: healthy, false positive, image 12 of sixth batch
    # actual: diseased prediction: diseased, true negative, image 0 of first batch
    # actual: healthy prediction: diseased, false negative, image 7 of first batch
    indclasses = [-1, -1, -1, -1]
    indclassesIM = [0, 0, 0, 0]
    # 6 images of plants with detected diseased
    inddetdis = []
    inddetdisIM = []
    # 6 images of plants with not detected diseased
    indnotdetdis = []
    indnotdetdisIM = []
    # Predict val dataset with final trained resnet, 1 = disease, 0 = no disease
    model.eval()
    pred, labels = [], []
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_ = model(X)
            _, y_label_ = torch.max(y_, 1)
            pred += y_label_.tolist()
            labels += y.tolist()
            Xdata, ydata, preddata = X, y.tolist(), y_label_.tolist()
            # get images and labels for images which get displayed
            for i in range(0, len(y)):
                if len(inddetdis) < 6 and preddata[i] == 1 and ydata[i] == 1:
                    inddetdis += [ydata[i]]
                    inddetdisIM.append(explain(model, Xdata[i], ydata[i]))
                if len(indnotdetdis) < 6 and preddata[i] == 0 and y[i] == 1:
                    indnotdetdis += [ydata[i]]
                    indnotdetdisIM.append(explain(model, Xdata[i], ydata[i]))
                if indclasses[2] == -1 and preddata[i] == 1 and ydata[i] == 1:
                    indclasses[2] = ydata[i]
                    indclassesIM[2] = explain(model, Xdata[i], ydata[i])
                if indclasses[3] == -1 and preddata[i] == 0 and ydata[i] == 1:
                    indclasses[3] = ydata[i]
                    indclassesIM[3] = explain(model, Xdata[i], ydata[i])
                if indclasses[0] == -1 and preddata[i] == 0 and ydata[i] == 0:
                    indclasses[0] = ydata[i]
                    indclassesIM[0] = explain(model, Xdata[i], ydata[i])
                if indclasses[1] == -1 and preddata[i] == 1 and ydata[i] == 0:
                    indclasses[1] = ydata[i]
                    indclassesIM[1] = explain(model, Xdata[i], ydata[i])

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

    # save the explainer images of the figures
    pathroot = './data/exp/'
    imageclass = []
    explainers = ['Original', 'saliency', 'IntegratedGradients', 'NoiseTunnel', 'GuidedGradCam', 'GradCam']
    numdis = min(len(inddetdis), len(indnotdetdis))

    # create explainer Image and save it in files
    if reexplain:

        # save images of explainer in data
        # save num diseased images which got detected
        for k in range(0, len(inddetdis)):
            for i in range(0, len(explainers)):
                inddetdisIM[k][i].savefig(pathroot + 'diseasedims/' + explainers[i] + 'diseased' + str(k) + '.png',
                                          bbox_inches='tight')

        # save images of explainer in data
        # save num diseased images which got not detected
        for k in range(0, len(inddetdis)):
            for i in range(0, len(explainers)):
                indnotdetdisIM[k][i].savefig(
                    pathroot + 'diseasedimsnotdetected/' + explainers[i] + 'diseased' + str(k) + '.png',
                    bbox_inches='tight')

        # save images of every class to compare
        for k in range(0, len(imageclass)):
            for i in range(0, len(explainers)):
                indclassesIM[k][i].savefig(pathroot + explainers[i] + imageclass[k] + '.png', bbox_inches='tight')

    # plot created explainer
    plotexplainer(pathroot, explainers, imageclass, numdis)


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
