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
from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift
from captum.attr import Saliency
from captum.attr import visualization as viz
from captum.attr import GuidedGradCam
from captum.attr._core.guided_grad_cam import LayerGradCam
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import feature

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")
retrain = False
reexplain = True
plot_for_image_id, plot_classes, plot_healthy, plot_diseased = False, False, False, True

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


# create explainers for given image
def explain(model, image, label):
    print("creating images")
    input = image.unsqueeze(0)
    input.requires_grad = True
    model.eval()
    c, h, w = image.shape
    # Edge detection of original input image
    org = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))
    org_img_edged = preprocessing.scale(np.array(org, dtype=float)[:, :, 1] / 255)
    org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
    # Compute the Canny filter for two values of sigma
    org_img_edged = feature.canny(org_img_edged, sigma=3)

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=label, **kwargs)
        return tensor_attributions

    def detect_edge(activation_map):
        # org = np.zeros((h, w), dtype=float) + org_img_edged
        # org = np.asarray(org_img_edged)[:, :, np.newaxis]
        fig, ax = plt.subplots()
        ax.imshow(org_img_edged, cmap=plt.cm.binary)
        ax.imshow(activation_map, cmap='viridis', vmin=np.min(activation_map), vmax=np.max(activation_map),
                  alpha=0.4)
        ax.tick_params(axis='both', which='both', length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.close('all')
        return fig

    def normalize(data):
        # consider only the positive values
        for i in range(h):
            for k in range(w):
                for j in range(c):
                    if data[i][k][j] < 0:
                        data[i][k][j] = 0
        # reshape to hxw
        d_img = data[:, :, 0] + data[:, :, 1] + data[:, :, 2]
        max = np.max(d_img)
        min = 0
        # normalize
        for i in range(h):
            for k in range(w):
                d_img[i][k] = (d_img[i][k] - min) / (max - min)
        return d_img

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    # saliency
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=label)
    grads = normalize(np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0)))

    # IntegratedGradients
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = normalize(np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)))

    # IntegratedGradients Noise Tunnel
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=5,
                                          # stdevs=0.2
                                          )

    attr_ig_nt = normalize(np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))

    # GuidedGradCam
    gc = GuidedGradCam(model, model.layer4)
    attr_gc = attribute_image_features(gc, input)
    attr_gc = normalize(np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))

    # GradCam Original Layer 4
    gco = LayerGradCam(model, model.layer4)
    attr_gco = attribute_image_features(gco, input)

    # GradCam
    att = attr_gco.squeeze(0).squeeze(0).cpu().detach().numpy()
    # gco_int = (att * 255).astype(np.uint8)
    gradcam = PImage.fromarray(att).resize((w, h), PImage.ANTIALIAS)
    np_gradcam = np.asarray(gradcam)
    print(np.max(np_gradcam))
    print(np.min(np_gradcam))


    f2 = detect_edge(grads)
    f3 = detect_edge(attr_ig)
    f4 = detect_edge(attr_ig_nt)
    f6 = detect_edge(attr_gc)
    f7 = f8 = detect_edge(np_gradcam)

    # original_image = detect_edge()
    # # Original Image
    f1, a1 = viz.visualize_image_attr(None, np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      method="original_image", use_pyplot=False)
    # # Overlayed Gradient Magnitudes saliency
    # f2, a2 = viz.visualize_image_attr(grads, original_image, sign="positive", method="blended_heat_map", use_pyplot=False)
    # # Overlayed Integrated Gradients
    # f3, a3 = viz.visualize_image_attr(attr_ig, original_image, sign="positive", method="blended_heat_map", use_pyplot=False)
    # # Overlayed Noise Tunnel
    # f4, a4 = viz.visualize_image_attr(attr_ig_nt, original_image, sign="positive",method="blended_heat_map", use_pyplot=False)
    #
    # # # DeepLift
    # # f5, a5 = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all",
    # #                                   # show_colorbar=True, title="Overlayed DeepLift"
    # #                                   )
    # # f5 = detect_edge(attr_dl)
    #
    # # GuidedGradCam
    # f6, a6 = viz.visualize_image_attr(attr_gc, original_image, sign="positive", method="blended_heat_map", show_colorbar=False, use_pyplot=False)
    #
    # # GradCam
    # f7, a7 = viz.visualize_image_attr(np_gradcam, original_image, sign="positive", method="blended_heat_map", show_colorbar=False, use_pyplot=False)
    #
    # # GradCam original image
    # f8, a8 = viz.visualize_image_attr(gradcam_orig, original_image, sign="absolute_value", method="blended_heat_map", show_colorbar=False, use_pyplot=False)
    # # Deeplift
    # dl = DeepLift(model)
    # attr_dl = attribute_image_features(dl, input)
    # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # print('Leaf is ', classes[predicted[ind]],
    #       'with a Probability of:', torch.max(F.softmax(outputs, 1)).item())
    return [f1, f2, f3, f4, f6, f7, f8]


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
        explained = explain(model.to(DEVICE), torch.from_numpy(image).to(DEVICE), label)
        for i in range(0, len(explainers)):
            directory = path_root + subpath + explainers[i] + image_id + '.png'
            explained[i].savefig(directory, bbox_inches='tight')


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
    fig = plt.figure(figsize=(6 * exnum + 2, 7 * number_images + 5))
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
        # labels: list of all labels
        # labels_ids: list of all labels with corresponding IDs as [[label, Id]...]
        # data: list of all images
        # data: list of all images with corresponding IDs as [[image, ID]...]
        # index_data: Pointer to image index in order of label Indecies
        # e.g. label with index i has its image data at Index index_data[i]
        # ids: list of all ids in order of labels
        self.labels_ids, self.labels, self.data_ids, self.data = self.load_images_for_labels(root, labels, mode=mode)
        self.transform = transform
        self.path = root
        self.index_data = []
        self.ids = []
        # index of the position of the images for corresponding label in labels_ids
        for k in self.labels_ids:
            self.ids.append(k[0])
            for i, s in enumerate(self.data_ids):
                if k[0] == s:
                    self.index_data += [i]
                    break

    def __getitem__(self, index):
        # return only 1 sample and label (according to "Index")
        # get label for ID
        label = self.labels[index]
        # # get corresponding Image for Index
        image = self.data[self.index_data[index]]
        return image, label

    def __len__(self):
        return len(self.labels)

    def get_id_by_index(self, index):
        try:
            return self.ids[index]
        except ValueError:
            print('No image in Dataset with id: ' + str(index))
            return None, None

    def get_by_id(self, ID):
        try:
            index = self.ids.index(ID)
            return self.__getitem__(index)
        except ValueError:
            print('image with id: ' + ID + ' not in dataset')
            return None, None

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
    batch_size = 20
    n_classes = 2
    N_EPOCHS = 50
    lr = 0.00025
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
        model = train(batch_size, n_classes, N_EPOCHS, lr, train_dl, val_dl)
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
    image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
    image_labels = []
    image_pred = []
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
            for i in image_ids:
                for k in range(1, 5):
                    evaluate_id(str(k) + '_' + i, all_ds, model, explainers, path_root, subpath_single_image + i + '/')

    print('creating comparator of explainer plots')
    if plot_for_image_id:
        # plot created explainer
        for i in image_ids:
            image_names = []
            for k in range(1, 5):
                image_names.append(str(k) + '_' + i)
            plot_single_explainer(path_root, subpath_single_image + i + '/', explainers, image_names,
                                  'Plant comparison over days of ID: ' + i)
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
