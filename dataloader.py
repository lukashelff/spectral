import pickle
from os import listdir
from typing import List, Any

from PIL import Image as PImage
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import ImageFolder, default_loader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from torchvision import models
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn import preprocessing
from utils import *

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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


# returns Array of tuples(String, (213, 255, 328)) with ID and memmap image data e.g. (3_Z2_1_0_1, memmap)
def load_images_for_labels(root_path, labels, mode):
    # loads a single image if the label entry exists
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
                    ims.append(data_all[k][:, :, [50, 88, 151]])
                    # ims.append(data_all[k][:, :, [24, 51, 118]])
                else:
                    ims.append(data_all[k])
                ids.append(i['id'].replace(',', '_'))
        return ids, ims

    # removes entries from labels where we dont have the image
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
                print("disk : " + str(i) + '_Z' + str(k))
                ids, im = load_image(root_path + str(i) + '_Z' + str(k) + '/segmented_leafs')
                loaded_images += im
                loaded_image_ids += ids
        else:
            for k in range(1, 19):
                print("disk : " + str(i) + '_Z' + str(k))
                if not (k == 16 and i == 4):
                    ids, im = load_image(root_path + str(i) + '_Z' + str(k) + '/segmented_leafs')
                    loaded_images += im
                    loaded_image_ids += ids
    label_ids, label_raw = sync_labels(loaded_image_ids)
    print("all images loaded")
    return label_ids, label_raw, loaded_image_ids, loaded_images


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
                    0 steht für gesund; 1 für Krank
        """

    def __init__(self, label, labels_ids, data, data_ids, root, transform=None):
        self.data = data
        self.data_ids = data_ids
        self.labels = label
        self.labels_ids = labels_ids
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


def is_float(string):
    try:
        return float(string)
    except ValueError:
        return False


def main():
    mode = 'rgb'
    # validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'

    print('loading training data')
    # load train dataset
    trainLabels, validLabels, all_labels = load_labels()
    labels_ids, labels_raw, image_ids, images_raw = load_images_for_labels(root, trainLabels, mode=mode)
    train_ds = Spectralloader(labels_raw, labels_ids, images_raw, image_ids, root)

    print('loading validation data')
    # load valid dataset
    labels_ids, labels_raw, image_ids, images_raw = load_images_for_labels(root, validLabels, mode=mode)
    val_ds = Spectralloader(labels_raw, labels_ids, images_raw, image_ids, root)

    batch_size = 2
    n_classes = 2
    N_EPOCHS = 2

    train_loss = np.zeros(N_EPOCHS)
    train_acc = np.zeros(N_EPOCHS)
    valid_loss = np.zeros(N_EPOCHS)
    valid_acc = np.zeros(N_EPOCHS)

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
        lr=0.0001,
        # momentum=0.9,
    )

    for epoch in range(N_EPOCHS):

        # Train
        model.train()

        total_loss, n_correct, n_samples = 0.0, 0, 0
        for batch_i, (X, y) in enumerate(train_dl):
            X, y = X.to(DEVICE), y.to(DEVICE)
            print(X.shape)
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

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} |"
            f"  train loss: {total_loss / n_samples:9.3f} |"
            f"  train acc:  {n_correct / n_samples * 100:9.3f}%"
        )
        train_loss[epoch] = total_loss / n_samples
        train_acc[epoch] = n_correct / n_samples * 100

        # Eval
        model.eval()

        total_loss, n_correct, n_samples = 0.0, 0, 0
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

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} |"
            f"  valid loss: {total_loss / n_samples:9.3f} |"
            f"  valid acc:  {n_correct / n_samples * 100:9.3f}%"
        )
        valid_loss[epoch] = total_loss / n_samples
        valid_acc[epoch] = n_correct / n_samples * 100

    # display rgb image
    def display_rgb(img):
        plt.imshow(img)
        plt.show()

    # display spectral image
    def display_spec(img, transpose=True):
        import spectral
        im = img[:, :, [50, 88, 151]]
        # im = img[:, :, [24, 51, 118]]
        plt.imshow(im)
        plt.show()

    display_rgb(images_raw[0])


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
