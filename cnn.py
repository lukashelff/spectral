from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import pickle
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import helpfunctions
import concurrent.futures as futures
from helpfunctions import *
from spectralloader import Spectralloader
import torchvision.datasets as t_datasets
import torch.utils.data as data
import torchvision.transforms as transforms


# from train_model import train_model

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


def get_model(DEVICE, n_classes, mode):
    if mode == 'imagenet':
        model = models.vgg16(pretrained=True)
        freeze_all(model.parameters())
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, n_classes)])  # Add our layer with n_classes outputs
        model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # resnet18 impl
        # model = models.resnet18(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # num_ftrs = model.fc.in_features
        # freeze_all(model.parameters())
        # model.fc = nn.Linear(num_ftrs, n_classes)
    if mode == 'plants':
        model = models.resnet18(pretrained=True)
        # use for LRP because AdaptiveAvgPool2d is not supported
        # model.avgpool = nn.MaxPool2d(kernel_size=7, stride=7, padding=0)
        freeze_all(model.parameters())
        model.fc = nn.Linear(512, n_classes)

    # print model
    # summary(model, (3, 255, 213), batch_size=20)
    # print(model)

    model.share_memory()
    model = model.to(DEVICE)
    return model


# trains and returns model for the given dataloader and computes graph acc, balanced acc and loss
def train(n_classes, N_EPOCHS, learning_rate, train_dl, val_dl, DEVICE, roar, cv_iteration, mode):
    lr_step_size = 7
    lr_gamma = 0.1
    optimizer_name = 'SGD'
    model_name = 'vgg'
    train_loss = np.zeros(N_EPOCHS)
    train_acc = np.zeros(N_EPOCHS)
    train_balanced_acc = np.zeros(N_EPOCHS)
    valid_loss = np.zeros(N_EPOCHS)
    valid_acc = np.zeros(N_EPOCHS)
    valid_balanced_acc = np.zeros(N_EPOCHS)

    model = get_model(DEVICE, n_classes, mode)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = None
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            get_trainable(model.parameters()),
            lr=learning_rate,
            # momentum=0.9,
        )
    else:
        optimizer = torch.optim.SGD(get_trainable(model.parameters()), lr=learning_rate, momentum=0.9)
    if mode == 'imagenet':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    text = 'training on ' + mode + ' DS with ' + roar + ' in cv it:' + str(cv_iteration)
    # exp_lr_scheduler = None
    with tqdm(total=N_EPOCHS, ncols=200) as progress:

        for epoch in range(N_EPOCHS):
            if exp_lr_scheduler is not None and epoch != 0:
                exp_lr_scheduler.step()
            # Train
            model.train()
            total_loss, n_correct, n_samples, pred, all_y = 0.0, 0, 0, [], []
            for batch_i, (X, y) in enumerate(train_dl):
                if X.shape != torch.Size([100, 3, 224, 224]):
                    print(X.shape)
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

            train_balanced_acc[epoch] = round(balanced_accuracy_score(all_y, pred) * 100, 2)
            train_loss[epoch] = round(total_loss / n_samples, 2)
            train_acc[epoch] = round(n_correct / n_samples * 100, 2)

            print(
                # f"Epoch {epoch + 1}/{N_EPOCHS} |"
                # f"  train loss: {train_loss[epoch]:9.3f} |"
                # f"  train acc:  {train_acc[epoch]:9.3f}% |"
                # f"  balanced acc:  {train_balanced_acc[epoch]:9.3f}%"

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

            valid_balanced_acc[epoch] = round(balanced_accuracy_score(all_y, pred) * 100, 2)
            valid_loss[epoch] = round(total_loss / n_samples, 2)
            valid_acc[epoch] = round(n_correct / n_samples * 100, 2)
            progress.update(1)
            progress.set_description(text + ' | current balanced acc: ' + str(valid_balanced_acc[epoch]) +
                                     f"Epoch {epoch + 1}/{N_EPOCHS} |"
                                     f"  train loss: {train_loss[epoch]:9.3f} |"
                                     f"  train acc:  {train_acc[epoch]:9.3f}% |"
                                     f"  valid loss: {valid_loss[epoch]:9.3f} |"
                                     f"  valid acc:  {valid_acc[epoch]:9.3f}% |"
                                     )
            progress.refresh()

            print(
                # f"Epoch {epoch + 1}/{N_EPOCHS} |"
                # f"  valid loss: {valid_loss[epoch]:9.3f} |"
                # f"  valid acc:  {valid_acc[epoch]:9.3f}% |"
                # f"  balanced acc:  {valid_balanced_acc[epoch]:9.3f}%"
            )

    # plot acc, balanced acc and loss
    if roar != "original":
        title = roar.replace('_', ' ') + ' image features removed'
    else:
        title = roar + ' model 0% removed'

    fig = plt.figure(num=None, figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(train_acc, color='skyblue', label='train acc')
    plt.plot(valid_acc, color='orange', label='valid_acc')
    plt.plot(train_balanced_acc, color='darkblue', label='train_balanced_acc')
    plt.plot(valid_balanced_acc, color='red', label='valid_balanced_acc')
    plt.title(title + 'with lr ' + str(learning_rate) + ', lr_step_size: ' + str(lr_step_size) + ', lr_gamma: ' + str(
        lr_gamma) +
              ', optimizer: ' + optimizer_name + ' on model: ' + model_name
              + '\nfinal bal acc: ' + str(round(valid_balanced_acc[N_EPOCHS - 1], 2)) + '%')
    plt.ylabel('model accuracy')
    plt.xlabel('training epoch')
    plt.axis([0, N_EPOCHS, 00, 100])
    plt.legend(loc='lower right')
    plt.savefig('./data/' + mode + '/' + 'plots/accuracy' + roar +
                '_lr_step_size_' + str(lr_step_size) +
                '_lr_gamma_' + str(lr_gamma) +
                '_optimizer_' + optimizer_name +
                '_model_' + model_name + '.png')
    plt.show()
    plt.plot(train_loss, color='red', label='train_loss')
    plt.plot(valid_loss, color='orange', label='valid_loss')
    plt.title('model loss ' + title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('./data/' + mode + '/' + 'plots/loss' + roar + '.png')
    plt.close(fig)
    path_values = './data/' + mode + '/' + 'plots/values/'
    # add accuracy values
    if not os.path.exists(path_values):
        os.makedirs(path_values)
    pickle.dump(round(valid_balanced_acc[N_EPOCHS - 1], 2),
                open(path_values + roar + '_cv_it_' + str(cv_iteration) + '.sav', 'wb'))
    return model


# ROAR remove and retrain
def train_roar_ds(path, roar_values, trained_roar_models, all_data, labels, batch_size, n_classes,
                  N_EPOCHS, lr, DEVICE, roar_explainers, sss, root, mode, cv_it_to_calc):
    # num_processes = len(roar_explainers)
    # pool = mp.Pool(1)
    for explainer in roar_explainers:
        cv_it = 0
        for train_index, test_index in sss.split(all_data, labels):
            if cv_it in cv_it_to_calc:
                train_labels = []
                valid_labels = []
                for i in train_index:
                    train_labels.append(all_data[i])
                for i in test_index:
                    valid_labels.append(all_data[i])
                print('loading training dataset')
                train_ds = Spectralloader(train_labels, root, mode)
                print('loading validation dataset')
                val_ds = Spectralloader(valid_labels, root, mode)
                path_root = path + explainer + '.pkl'
                with open(path_root, 'rb') as f:
                    mask = pickle.load(f)
                    processes = []
                    for i in roar_values:
                        # processes.append((i, mask, DEVICE, explainer, val_ds_org, train_ds_org,
                        #                                    batch_size, n_classes, N_EPOCHS, lr, trained_roar_models,))
                        # train_parallel(i, mask, DEVICE, explainer, val_ds_org, train_ds_org, batch_size, n_classes, N_EPOCHS, lr, trained_roar_models)

                        p = mp.Process(target=train_parallel, args=(i, mask, DEVICE, explainer, val_ds, train_ds,
                                                                    batch_size, n_classes, N_EPOCHS, lr,
                                                                    trained_roar_models, cv_it, mode))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
            cv_it += 1
        torch.cuda.empty_cache()


# pool.close()
# pool.join()

# print('------------------------------------------------------------')
# print('removing ' + str(i) + ' % of the image features & train after ' + explainer)  #
# val_ds = deepcopy(val_ds_org)
# print('applying ROAR heatmap to validation DS')
# val_ds.apply_roar(i, mask, DEVICE, explainer)
# val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
# # print example image
# im, label = val_ds.get_by_id('4_Z15_1_1_0')
# path = './data/exp/pred_img_example/'
# name = explainer + 'ROAR' + str(i)
# # display the modified image and save to pred images in data/exp/pred_img_example
# # display_rgb(im, 'image with ' + str(i) + '% of ' + explainer + ' values removed ', path, name)
# train_ds = deepcopy(train_ds_org)
# print('applying ROAR heatmap to training DS')
# train_ds.apply_roar(i, mask, DEVICE, explainer)
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
# print('training on ROAR DS, ' + str(i) + ' % removed')
# model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE,
#               str(i) + '%_of_' + explainer)
# print('saving roar model')
# torch.save(model.state_dict(), trained_roar_models + '_' + explainer + '_' + str(i) + '.pt')


## train for each created split
def train_cross_val(sss, all_data, labels, root, mode, batch_size, n_classes, N_EPOCHS, lr, DEVICE,
                    original_trained_model, cv_it_to_calc):
    cv_it = 0
    processes = []

    for train_index, test_index in sss.split(np.zeros(len(labels)), labels):
        if cv_it in cv_it_to_calc:
            train_data = [all_data[i] for i in train_index]
            valid_data = [all_data[i] for i in test_index]
            print('loading validation dataset')
            val_ds = Spectralloader(valid_data, root, mode)
            print('loading training dataset')
            train_ds = Spectralloader(train_data, root, mode)
            im, label = train_ds.__getitem__(0)
            path = './data/' + mode + '/' + 'exp/pred_img_example/'
            # display the modified image and save to pred images in data/exp/pred_img_example
            # show_image(im, 'original image ')

            train_parallel(0, None, DEVICE, 'original', val_ds, train_ds, batch_size, n_classes, N_EPOCHS, lr,
                           original_trained_model, cv_it, mode)

            # p = mp.Process(target=train_parallel, args=(0, None, DEVICE, 'original', val_ds, train_ds,
            #                                             batch_size, n_classes, N_EPOCHS, lr, original_trained_model, cv_it, mode))

        cv_it += 1
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()


def train_parallel(roar_val, mask, DEVICE, explainer, val_ds_org, train_ds_org, batch_size, n_classes, N_EPOCHS, lr,
                   trained_roar_models, cv_it, mode):
    if explainer == 'original':
        train_dl = DataLoader(train_ds_org, batch_size=batch_size, shuffle=True, num_workers=4, )
        val_dl = DataLoader(val_ds_org, batch_size=batch_size, shuffle=False, num_workers=4, )
        original_model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, "original", cv_it, mode)
        torch.save(original_model.state_dict(), trained_roar_models)

    else:
        val_ds = deepcopy(val_ds_org)
        train_ds = deepcopy(train_ds_org)
        # processes = [(val_ds, i, mask, DEVICE, explainer), (train_ds, i, mask, DEVICE, explainer)]
        # p1 = mp.Process(target=apply_parallel, args=(val_ds, i, mask, DEVICE, explainer))
        # p2 = mp.Process(target=apply_parallel, args=(train_ds, i, mask, DEVICE, explainer))
        train_ds.apply_roar(roar_val, mask, DEVICE, explainer)
        val_ds.apply_roar(roar_val, mask, DEVICE, explainer)
        # p1.start()
        # p2.start()
        # p1.join()
        # p2.join()
        # with futures.ProcessPoolExecutor(max_workers=5) as ex:
        #     fs = ex.map(train_parallel, processes)
        # futures.wait(fs)
        # print example image
        im, label = train_ds.__getitem__(0)
        path = './data/' + mode + '/' + 'exp/pred_img_example/'
        name = 'ROAR_' + str(roar_val) + '_of_' + explainer + '.jpeg'
        # display the modified image and save to pred images in data/exp/pred_img_example
        display_rgb(im, 'image with ' + str(roar_val) + '% of ' + explainer + 'values removed', path, name)

        # create Dataloaders
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
        # print('training on DS with ' + str(i) + ' % of ' + explainer + ' image features removed')
        model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, str(roar_val) + '%_of_' + explainer, cv_it,
                      mode)
        torch.save(model.state_dict(), trained_roar_models + '_' + explainer + '_' + str(roar_val) + '.pt')


def apply_parallel(ds, i, mask, DEVICE, explainer):
    ds.apply_roar(i, mask, DEVICE, explainer)


def train_imagenet(N_EPOCHS, lr, batch_size, DEVICE, mode):
    data_dir = './data/imagenet/tiny-imagenet-200/'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }

    image_datasets = {x: t_datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    train(200, N_EPOCHS, lr, dataloaders['train'], dataloaders['val'], DEVICE, 'original', 0, mode)
