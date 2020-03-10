import torch
import tqdm as tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import tqdm
import sys
import multiprocessing as mp
import helpfunctions


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
        # data_ids: list of all corresponding IDs for image order as [ID...]
        # index_data: Pointer to image index in order of label Indices
        # e.g. label with index i has its image data at Index index_data[i]
        # ids: list of all ids in order of labels
        self.labels_ids, self.labels, self.data_ids, self.data = self.load_images_for_labels(root, labels, mode=mode)
        self.dict_data = {}
        self.transform = transform
        self.path = root
        self.index_data = []
        self.ids = []
        # index of the position of the images for corresponding label in labels_ids
        for k in self.labels_ids:
            self.ids.append(k[0])
            for i, s in enumerate(self.data_ids):
                if k[0] == s:
                    self.dict_data.update({s: (self.data[i], k[1])})
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

    # update value in dataset with the new specified value
    def update_data(self, id, val):

        for c, value in enumerate(self.data_ids):
            if value == id:
                self.data[c] = val

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
        return label_ids, label_raw, loaded_image_ids, loaded_images

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
        #parallel execution not working
        # pool = mp.Pool(20)
        # for d in range(0, length):
        #     id = self.get_id_by_index(d)
        #     pool.apply_async(self.parallel_roar, (percentage, masks, id, "mean", explainer))
        # pool.close()
        # pool.join()
            # r = list(tqdm.tqdm(pool.imap_unordered(self.apply_roar_single_image, data), total=length, desc=text))
        with tqdm.tqdm(total=length, desc=text) as progress:
            for d in range(0, length):
                id = self.get_id_by_index(d)
                progress.update(1)
                self.apply_roar_single_image(percentage, masks, id, "mean", explainer)

    def parallel_roar(self, percentage, masks, id, mean, explainer):
        self.apply_roar_single_image(percentage, masks, id, "mean", explainer)


