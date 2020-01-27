import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import pickle
import os

from helpfunctions import display_rgb
from spectralloader import Spectralloader


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
def train(n_classes, N_EPOCHS, learning_rate, train_dl, val_dl, DEVICE, roar):
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
    plt.title('model accuracy ' + roar + ', final balanced accuracy: ' + str(
        round(valid_balanced_acc[N_EPOCHS - 1], 2)) + '%')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('./data/plots/accuracy' + roar + '.png')
    plt.show()
    plt.plot(train_loss, color='red', label='train_loss')
    plt.plot(valid_loss, color='orange', label='valid_loss')
    plt.title('model loss ' + roar)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('./data/plots/loss' + roar + '.png')
    plt.show()
    path_values = './data/plots/values/'
    if not os.path.exists(path_values):
        os.makedirs(path_values)
    pickle.dump(round(valid_balanced_acc[N_EPOCHS - 1], 2), open(path_values + roar + '.sav', 'wb'))

    return model


def train_roar_ds(path_root, subpath_heapmaps, root, roar_values, filename_roar, valid_labels, train_labels, batch_size,
                  n_classes, N_EPOCHS, lr, mode, DEVICE):
    with open(path_root + subpath_heapmaps, 'rb') as f:
        mask = pickle.load(f)
        for i in roar_values:
            print('training with roar dataset ' + str(i) + ' % of the image features removed')  #
            print('loading validation DS')
            val_ds = Spectralloader(valid_labels, root, mode)
            print('applying ROAR to validation DS')
            val_ds.apply_roar(i, mask, DEVICE)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
            # print example image
            im, label = val_ds.get_by_id('4_Z15_1_1_0')
            display_rgb(im, 'example image roar:' + str(i))
            print('loading training data')
            train_ds = Spectralloader(train_labels, root, mode)
            print('applying ROAR to training DS')
            train_ds.apply_roar(i, mask, DEVICE)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
            print('training on dS with ROAR= ' + str(i) + ' % removed')
            model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, str(i) + "%removed")
            print('saving roar model')
            pickle.dump(model, open(filename_roar + str(i) + '.sav', 'wb'))
    acc_vals = []
    for i in roar_values:
        path = './data/plots/values/' + str(i) + "%removed" + '.sav'
        val = pickle.load(open(path, 'rb'))
        acc_vals.append(val)
    plt.plot(roar_values, acc_vals, 'ro')
    plt.title('development of accuracy by increasing ROAR value')
    plt.xlabel('ROAR value: % removed from image')
    plt.ylabel('accuracy')
    plt.axis([0, 100, 0, 100])
    plt.savefig('./data/plots/accuracy_roar_comparison')
    plt.show()
