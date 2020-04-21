import glob
import os
import mmap
import tqdm as tqdm
from torch.utils.data import Dataset
import pickle
import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp
import cv2
from shutil import move
from os import rmdir
from sys import getsizeof
import gc
import resource
import torch

from torchvision import transforms
import torchvision.datasets as t_datasets


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

    def __init__(self, ids_and_labels, root, mode, transform=None):
        # Parameter:
        # ids_and_labels: list of all IDs with their corresponding labels
        # Variables:
        # mode: either 'imagenet' or 'plants' determine the correct DS to load from
        # ids: list of all ids in order of data
        # data: dictionary of all IDs with their corresponding images and label
        #  data[id]['image'] = image, data[id]['label'] = label
        gc.enable()
        self.mode = mode
        self.data, self.ids = self.load_images_for_labels(root, ids_and_labels)
        gc.disable()

    def __getitem__(self, index):
        # return only 1 sample and label according to "Index"
        id = self.get_id_by_index(index)
        image, label = self.get_by_id(id)
        return image, label

    def __len__(self):
        return len(self.ids)

    # update value in dataset with the new specified value
    def update_data(self, id, val):
        try:
            self.data[id]['image'] = val
        except ValueError:
            print('image with id: ' + id + ' not in dataset')

    def get_id_by_index(self, index):
        try:
            return self.ids[index]
        except ValueError:
            print('Index out of bound: ' + str(index))
            return None

    def get_by_id(self, id):
        try:
            image, label = self.data[id]['image'], self.data[id]['label']
            if self.mode == 'imagenet':
                image = image
            return image, label
        except ValueError:
            print('image with id: ' + id + ' not in dataset')
            return None, None

    # returns an Array of IDs and a dictionary of all IDs with their corresponding images and label
    def load_images_for_labels(self, root_path, labels):
        data = {}
        ids = []

        # add image with corresponding label and id to the DS
        def add_to_data(image, id):
            for (k, label) in labels:
                if k == id:
                    data[id] = {}
                    data[id]['image'] = image
                    data[id]['label'] = label
                    ids.append(k)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('Memory usage in KB after: ' + str(mem))

        if self.mode == 'imagenet':
            data_transforms = {
                'train': transforms.Compose([
                    # transforms.RandomRotation(20),
                    # transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
            }
            data_dir = 'data/' + self.mode + '/' + 'tiny-imagenet-200'
            image_datasets = {x: t_datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                              for x in ['train', 'val']}
            len_train = image_datasets['train'].__len__()
            len_val = image_datasets['val'].__len__()
            text = 'loading images of ' + self.mode + ' DS: '
            with tqdm(total=len_train + len_val, desc=text,
                      ncols=180) as progress:
                for i in range(len_train):
                    image, label = image_datasets['train'].__getitem__(i)
                    add_to_data(image, str(i))
                    progress.update(1)
                for i in range(len_val):
                    image, label = image_datasets['val'].__getitem__(i)
                    add_to_data(image, str(i + len_train))
                    progress.update(1)
        else:
            # loads all the images have existing entry labels in the plant DS
            def load_image(path):
                dict = pickle.load(open(path + '/data.p', 'rb'))
                shape = dict['memmap_shape']
                samples = dict['samples']
                data_all = np.memmap(path + '/memmap.dat', mode='r', shape=shape, dtype='float32')
                for k, i in enumerate(samples):
                    # only add if we have a label for the image
                    data = np.transpose(data_all[k][:, :, [50, 88, 151]], (2, 0, 1))
                    add_to_data(data, i['id'].replace(',', '_'))
                    # elif mode == 'spec': reserved for spectral implementation
                    #     add_to_data(data_all[k].reshape(3, 255, 213), i['id'].replace(',', '_'))

            with tqdm(total=67) as progress:

                for i in range(1, 5):
                    if i == 1:
                        for k in range(1, 14):
                            progress.update(1)
                            load_image(root_path + str(i) + '_Z' + str(k) + '/segmented_leafs')
                    else:
                        for k in range(1, 19):
                            progress.update(1)
                            if not (k == 16 and i == 4):
                                load_image(root_path + str(i) + '_Z' + str(k) + '/segmented_leafs')
        print('size of data ' + str(getsizeof(data)))
        return data, ids

    def apply_roar_single_image(self, percentage, masks, id, new_val, explainer):
        im = None
        try:
            im, label = self.get_by_id(id)
        except ValueError:
            print('No roar img for id: ' + id)
        if im is not None:
            mean = np.mean(im)
            mask = masks[id]
            # only take percentile of values with duplicated zeros deleted
            mask_flat = mask.flatten()
            percentile = np.percentile(mask_flat, 100 - percentage)
            c, h, w = im.shape
            val = im
            bigger = 0
            indices_of_same_values = []
            for i in range(0, w):
                for j in range(0, h):
                    if mask[j][i] > percentile:
                        bigger += 1
                        if new_val == "mean":
                            val[0][j][i] = mean
                            val[1][j][i] = mean
                            val[2][j][i] = mean
                        else:
                            val[0][j][i] = 238 / 255
                            val[1][j][i] = 173 / 255
                            val[2][j][i] = 14 / 255
                    if mask[j][i] == percentile:
                        indices_of_same_values.append([j, i])
            if len(indices_of_same_values) > 5:
                missing = max(int(0.01 * percentage * w * h - bigger), 0)
                selection = random.sample(indices_of_same_values, missing)
                for i in selection:
                    if new_val == "mean":
                        val[0][i[0]][i[1]] = mean
                        val[1][i[0]][i[1]] = mean
                        val[2][i[0]][i[1]] = mean
                    else:
                        val[0][i[0]][i[1]] = 238 / 255
                        val[1][i[0]][i[1]] = 173 / 255
                        val[2][i[0]][i[1]] = 14 / 255
            self.update_data(id, val)

    # apply the roar to the dataset
    # given percentage of the values get removed from the dataset
    def apply_roar(self, percentage, masks, DEVICE, explainer):
        length = self.__len__()
        text = 'removing ' + str(percentage) + '% of ' + explainer
        # parallel execution not working
        # pool = mp.Pool(20)
        # for d in range(0, length):
        #     id = self.get_id_by_index(d)
        #     pool.apply_async(self.parallel_roar, (percentage, masks, id, "mean", explainer))
        # pool.close()
        # pool.join()
        # r = list(tqdm.tqdm(pool.imap_unordered(self.apply_roar_single_image, data), total=length, desc=text))
        with tqdm(total=length, desc=text) as progress:
            for d in range(0, length):
                id = self.get_id_by_index(d)
                progress.update(1)
                self.apply_roar_single_image(percentage, masks, id, "mean", explainer)

    def parallel_roar(self, percentage, masks, id, mean, explainer):
        self.apply_roar_single_image(percentage, masks, id, "mean", explainer)


# returns Array of tuples(String, int) with ID and disease information 0 disease/ 1 healthy e.g. (3_Z2_1_0_1, 0)
# returns Array of tuples(String, int) with ID and class information e.g. (test_9925.JPEG n01910747)
# train Array of tuples(String, int) with ID and class information
# valid Array of tuples(String, int) with ID and class information
# all_data Array of tuples(String, int) with ID and class information
# all_labels Array of int with class information
def load_labels(mode):
    valid = []
    train = []
    all_labels = []

    # load all imagenet labels use the index as the id as a unique identifier
    if mode == 'imagenet':
        data_dir = 'data/' + mode + '/' + 'tiny-imagenet-200'
        image_datasets = {x: t_datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
        train_labels = image_datasets['train'].targets
        train = [(str(c), i) for c, i in enumerate(train_labels)]
        val_labels = image_datasets['val'].targets
        valid = [(str(len(train_labels) + c), len(train_labels) + i) for c, i in enumerate(val_labels)]

        return None, None, train + valid, train_labels + val_labels
    else:
        # mp.set_start_method('spawn')
        path_test = 'data/' + mode + '/' + 'test_fileids.txt'
        path_train = 'data/' + mode + '/' + 'train_fileids.txt'
        valid_s = open(path_test, 'r').readlines()
        train_s = open(path_train, 'r').readlines()
        for i in valid_s:
            data = i.split(';')
            valid.append((data[0], int(data[1])))
            all_labels.append(int(data[1]))
        for i in train_s:
            data = i.split(';')
            train.append((data[0], int(data[1])))
            all_labels.append(int(data[1]))
    return train, valid, train + valid, all_labels


def upscale_imagenet():
    target_size = 224
    all_images = glob.glob('data/imagenet/tiny-imagenet-200/*/*/*/*')

    def resize_img(image_path, size):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_path, img)

    text = 'upscaling images from imagenet'
    with tqdm(total=len(all_images), desc=text) as progress:
        for image in all_images:
            resize_img(image, target_size)
            progress.update(1)


def val_format():
    target_folder = './data/imagenet/tiny-imagenet-200/val/'
    test_folder = './data/imagenet/tiny-imagenet-200/test/'

    val_dict = {}
    with open('./data/imagenet/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob('./data/imagenet/tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')
        if not os.path.exists(test_folder + str(folder)):
            os.mkdir(test_folder + str(folder))
            os.mkdir(test_folder + str(folder) + '/images')

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if len(glob.glob(target_folder + str(folder) + '/images/*')) < 25:
            dest = target_folder + str(folder) + '/images/' + str(file)
        else:
            dest = test_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir('./data/imagenet/tiny-imagenet-200/val/images')
