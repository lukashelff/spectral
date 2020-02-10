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
    if roar != "original":
        title = roar.replace('_', ' ') + ' image features removed'
    else:
        title = roar + ' model 0% removed'
    plt.plot(train_acc, color='skyblue', label='train acc')
    plt.plot(valid_acc, color='orange', label='valid_acc')
    plt.plot(train_balanced_acc, color='darkblue', label='train_balanced_acc')
    plt.plot(valid_balanced_acc, color='red', label='valid_balanced_acc')
    plt.title(title + ',\nfinal bal acc: ' + str(
        round(valid_balanced_acc[N_EPOCHS - 1], 2)) + '%')
    plt.ylabel('model accuracy')
    plt.xlabel('training epoch')
    plt.axis([0, N_EPOCHS, 50, 100])
    plt.legend(loc='lower right')
    plt.savefig('./data/plots/accuracy' + roar + '.png')
    plt.show()
    plt.plot(train_loss, color='red', label='train_loss')
    plt.plot(valid_loss, color='orange', label='valid_loss')
    plt.title('model loss ' + title)
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


# ROAR remove and retrain
def train_roar_ds(path_root, root, roar_values, trained_roar_models, valid_labels, train_labels, batch_size, n_classes,
                  N_EPOCHS, lr, mode, DEVICE, explainer):
    with open(path_root, 'rb') as f:
        mask = pickle.load(f)
        for i in roar_values:
            print('------------------------------------------------------------')
            print('removing ' + str(i) + ' % of the image features & train after ' + explainer)  #
            val_ds = Spectralloader(valid_labels, root, mode)
            print('applying ROAR heapmap to validation DS')
            val_ds.apply_roar(i, mask, DEVICE)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
            # print example image
            im, label = val_ds.get_by_id('4_Z15_1_1_0')
            path = './data/exp/pred_img_example/'
            name = explainer + 'ROAR' + str(i)
            # display the modified image and save to pred images in data/exp/pred_img_example
            display_rgb(im, 'image with ' + str(i) + '% of ' + explainer + ' values removed ', path, name)
            train_ds = Spectralloader(train_labels, root, mode)
            print('applying ROAR heapmap to training DS')
            train_ds.apply_roar(i, mask, DEVICE)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
            print('training on ROAR DS, ' + str(i) + ' % removed')
            model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE,
                          str(i) + '%_of_' + explainer)
            print('saving roar model')
            # pickle.dump(model, open(trained_roar_models + '_' + explainer + '_' + str(i) + '.sav', 'wb'))
            torch.save(model.state_dict(), trained_roar_models + '_' + explainer + '_' + str(i) + '.pt')
