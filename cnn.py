from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as t_datasets
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib
from matplotlib import rc
import matplotlib.font_manager
import matplotlib as mpl

from helpfunctions import *
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


def get_model(DEVICE, n_classes, mode, model):
    if model == 'VGG':
        # model = models.vgg16(pretrained=True)
        # test_ipt = Variable(torch.zeros(1, 3, 64, 64))
        # test_out = models.vgg.features(test_ipt)
        # n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)

        model = models.vgg16(pretrained=True)
        freeze_all(model.parameters())
        num_features = model.classifier[6].in_features
        # adaptive average pooling is need if input size of image does not match 224x224
        model.avgpool = nn.MaxPool2d(1, )
        model.classifier[6] = nn.Linear(num_features, n_classes)
        # features = list(model.classifier.children())[:-1]  # Remove last layer and first
        # features.extend([nn.Linear(num_features, n_classes)])  # Add our layer with n_classes outputs
        # model.classifier = nn.Sequential(*features)  # Replace the model classifier

    if model == 'ResNet':
        model = models.resnet18(pretrained=True)
        # use for LRP because AdaptiveAvgPool2d is not supported
        # model.avgpool = nn.MaxPool2d(kernel_size=7, stride=7, padding=0)
        freeze_all(model.parameters())
        model.fc = nn.Linear(512, n_classes)

        # resnet18 impl without adaptive averagepool
        # model = models.resnet18(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # num_ftrs = model.fc.in_features
        # freeze_all(model.parameters())
        # model.fc = nn.Linear(num_ftrs, n_classes)

    # print model
    # summary(model, (3, 255, 213), batch_size=20)
    # print(model)True

    model.share_memory()
    model = model.to(DEVICE)
    return model


# trains and returns model for the given dataloader and computes graph acc, balanced acc and loss
def train(n_classes, N_EPOCHS, learning_rate, train_dl, val_dl, DEVICE, roar, cv_iteration, mode, model_type):
    lr_step_size = 7
    lr_gamma = 0.1
    scheduler_a = '_scheduler'
    if mode == 'plants':
        scheduler_a = '_no_scheduler'
        lr_step_size = 40
        lr_gamma = 0.1
    optimizer_name = 'adam'
    # set scheduler to use scheduler
    ####################
    ################
    # print(model_name)
    train_loss = np.zeros(N_EPOCHS)
    train_acc = np.zeros(N_EPOCHS)
    train_balanced_acc = np.zeros(N_EPOCHS)
    valid_loss = np.zeros(N_EPOCHS)
    valid_acc = np.zeros(N_EPOCHS)
    valid_balanced_acc = np.zeros(N_EPOCHS)
    valid_tpr = np.zeros(N_EPOCHS)
    valid_tnr = np.zeros(N_EPOCHS)

    model = get_model(DEVICE, n_classes, mode, model_type)
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
    if scheduler_a == '_scheduler':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        scheduler_a += '_lr_step_size_' + str(lr_step_size) + '_lr_gamma_' + str(lr_gamma)
    save_name = (
            '_pretraining' +
            '_no_normalization' +
            '_resize' +
            '_lr_' + str(learning_rate) +
            # '_pixel_64' +
            scheduler_a +
            '_optimizer_' + optimizer_name +
            '_model_' + model_type +
            roar
    )
    print(save_name)
    text = 'training on ' + mode + ' DS with ' + roar + ' in cv it:' + str(cv_iteration)
    with tqdm(total=N_EPOCHS, ncols=180) as progress:

        for epoch in range(N_EPOCHS):
            if epoch == 0:
                progress.set_description(text)
                progress.refresh()
            if exp_lr_scheduler is not None and epoch != 0:
                exp_lr_scheduler.step()
            # Train
            model.train()
            total_loss, n_correct, n_samples, pred, all_y = 0.0, 0, 0, [], []
            for batch_i, (X, y) in enumerate(train_dl):
                # print('current batch: ' + str(batch_i))
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
            # valid_tpr = classification_report(all_y, pred)
            # valid_tnr =
            progress.update(1)
            progress.set_description(text + ' | ' +
                                     # f"  train loss: {train_loss[epoch]:9.3f} |"
                                     f"  train acc:  {train_balanced_acc[epoch]:9.3f}% |"
                                     # f"  valid loss: {valid_loss[epoch]:9.3f} |"
                                     f"  valid acc:  {valid_balanced_acc[epoch]:9.3f}% |"
                                     )
            progress.refresh()

            # print(
            # f"Epoch {epoch + 1}/{N_EPOCHS} |"
            # f"  valid loss: {valid_loss[epoch]:9.3f} |"
            # f"  valid acc:  {valid_acc[epoch]:9.3f}% |"
            # f"  balanced acc:  {valid_balanced_acc[epoch]:9.3f}%"
            # )

    # plot acc, balanced acc and loss
    if roar != "original":
        title = roar.replace('_', ' ') + ' image features removed'
    else:
        title = roar + ' model 0% removed'
    font = {
            'size': 15,
            # 'family': 'serif',
            # 'serif': ['Computer Modern']
            'family': 'sans-serif',
            'serif': ['Computer Modern Sans serif']
            }

    rc('font', **font)
    # rc('text', usetex=True)
    fig = plt.figure(num=None, figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
    if mode != 'imagenet':
        plt.plot(train_acc, color='skyblue', label='train acc')
        plt.plot(valid_acc, color='orange', label='valid_acc')
    plt.plot(train_balanced_acc, color='darkblue', label='train_balanced_acc')
    plt.plot(valid_balanced_acc, color='red', label='valid_balanced_acc')
    # plt.title(title +
    #           ' with lr ' + str(learning_rate) + ', lr_step_size: ' +
    #           str(lr_step_size) + ', lr_gamma: ' + str(lr_gamma) +
    #           ', optimizer: ' + optimizer_name + ' on model: ' + model_type +
    #           '\nfinal bal acc: ' + str(round(valid_balanced_acc[N_EPOCHS - 1], 2)) + '%')
    plt.title(model_type, size=25)
    plt.ylabel('model accuracy', size=20)
    plt.xlabel('training epoch', size=20)
    min_acc = int(min(np.concatenate((train_balanced_acc, valid_balanced_acc, train_acc, valid_acc))) / 10) * 10
    max_acc = (1 + int(max(np.concatenate((train_balanced_acc, valid_balanced_acc, train_acc, valid_acc))) / 10)) * 10
    plt.axis([0, N_EPOCHS - 1, min_acc, max_acc])
    plt.legend(loc='lower right')

    plt.savefig('./data/' + mode + '/' + 'plots/accuracy' +
                save_name +
                '.png',
                dpi=200
                )
    plt.close(fig)
    font = {
            'size': 10,
            # 'family': 'serif',
            # 'serif': ['Computer Modern']
            'family': 'sans-serif',
            'serif': ['Computer Modern Sans serif']
            }
    rc('font', **font)
    plt.show()
    plt.plot(train_loss, color='red', label='train_loss')
    plt.plot(valid_loss, color='orange', label='valid_loss')
    plt.title('loss', size=20)
    plt.ylabel('loss', size=15)
    plt.xlabel('epoch', size=15)
    plt.legend(loc='lower right')
    plt.savefig('./data/' + mode + '/' + 'plots/loss' +
                save_name +
                '.png',
                dpi=200
                )
    plt.close(fig)
    path_values = './data/' + mode + '/' + 'plots/values/' + model_type + '/'
    # add accuracy values
    if not os.path.exists(path_values):
        os.makedirs(path_values)
    pickle.dump(round(valid_balanced_acc[N_EPOCHS - 1], 2),
                open(path_values + roar + '_cv_it_' + str(cv_iteration) + '.sav', 'wb'))
    return model


# ROAR remove and retrain
def train_roar_ds(path, roar_values, trained_roar_models, all_data, labels, batch_size, n_classes,
                  N_EPOCHS, lr, DEVICE, roar_explainers, sss, root, mode, cv_it_to_calc, model_type):
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
                train_ds = Spectralloader(train_labels, root, mode, 'train')
                print('loading validation dataset')
                val_ds = Spectralloader(valid_labels, root, mode, 'val')
                id = train_ds.get_id_by_index(10)
                im, label = train_ds.get_original_by_id(id)
                show_image(im, 'roar')
                path_mask = path + explainer + '.pkl'
                processes = []
                for i in roar_values:
                    # processes.append((i, mask, DEVICE, explainer, val_ds_org, train_ds_org,
                    #                                    batch_size, n_classes, N_EPOCHS, lr, trained_roar_models,))
                    train_parallel(i, path, DEVICE, explainer, val_ds, train_ds, batch_size, n_classes, N_EPOCHS,
                                   lr, trained_roar_models, cv_it, mode, model_type)

                    #     p = mp.Process(target=train_parallel, args=(i, mask, DEVICE, explainer, val_ds, train_ds,
                    #                                                 batch_size, n_classes, N_EPOCHS, lr,
                    #                                                 trained_roar_models, cv_it, mode))
                    #     p.start()
                    #     processes.append(p)
                    # for p in processes:
                    #     p.join()
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
                    original_trained_model, cv_it_to_calc, model_type):
    cv_it = 0
    processes = []

    for train_index, test_index in sss.split(np.zeros(len(labels)), labels):
        if cv_it in cv_it_to_calc:
            train_data = [all_data[i] for i in train_index]
            valid_data = [all_data[i] for i in test_index]
            print('loading validation dataset')
            val_ds = Spectralloader(valid_data, root, mode, 'val')
            print('loading training dataset')
            train_ds = Spectralloader(train_data, root, mode, 'train')
            im, label = train_ds.__getitem__(0)
            path = './data/' + mode + '/' + 'exp/pred_img_example/'
            name = 'original_image.jpeg'
            # display_rgb(im, 'original image no values removed', path, name)
            # display the modified image and save to pred images in data/exp/pred_img_example
            show_image(im, 'original image ')

            train_parallel(0, None, DEVICE, 'original', val_ds, train_ds, batch_size, n_classes, N_EPOCHS, lr,
                           original_trained_model, cv_it, mode, model_type)

            # p = mp.Process(target=train_parallel, args=(0, None, DEVICE, 'original', val_ds, train_ds, batch_size, n_classes, N_EPOCHS, lr, original_trained_model, cv_it, mode))

        cv_it += 1
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()


def train_parallel(roar_val, path_mask, DEVICE, explainer, val_ds, train_ds, batch_size, n_classes, N_EPOCHS, lr,
                   trained_roar_models, cv_it, mode, model_type):
    if explainer == 'original':
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
        original_model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, "original", cv_it, mode, model_type)
        torch.save(original_model.state_dict(), trained_roar_models)

    else:
        val_ds = deepcopy(val_ds)
        train_ds = deepcopy(train_ds)
        # processes = [(val_ds, i, mask, DEVICE, explainer), (train_ds, i, mask, DEVICE, explainer)]
        # p1 = mp.Process(target=apply_parallel, args=(val_ds, i, mask, DEVICE, explainer))
        # p2 = mp.Process(target=apply_parallel, args=(train_ds, i, mask, DEVICE, explainer))
        train_ds.apply_roar(roar_val, path_mask, DEVICE, explainer, model_type)
        val_ds.apply_roar(roar_val, path_mask, DEVICE, explainer, model_type)

        # p1.start()
        # p2.start()
        # p1.join()
        # p2.join()
        # with futures.ProcessPoolExecutor(max_workers=5) as ex:
        #     fs = ex.map(train_parallel, processes)
        # futures.wait(fs)
        # print example image
        id = train_ds.get_id_by_index(10)
        im, label = train_ds.get_original_by_id(id)
        path = './data/' + mode + '/' + 'exp/pred_img_example/' + model_type
        name = '/ROAR_' + str(roar_val) + '_of_' + explainer + '.jpeg'
        # display the modified image and save to pred images in data/exp/pred_img_example
        display_rgb(im, 'image with ' + str(roar_val) + '% of ' + explainer + 'values removed', path, name)

        # create Dataloaders
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, )
        # print('training on DS with ' + str(i) + ' % of ' + explainer + ' image features removed')
        model = train(n_classes, N_EPOCHS, lr, train_dl, val_dl, DEVICE, str(roar_val) + '%_of_' + explainer, cv_it,
                      mode, model_type)
        torch.save(model.state_dict(), trained_roar_models + '_' + explainer + '_' + str(roar_val) + '.pt')


def apply_parallel(ds, i, mask, DEVICE, explainer, model_type):
    ds.apply_roar(i, mask, DEVICE, explainer, model_type)


def eval_model(model, N_EPOCHS, lr, batch_size, DEVICE, mode):
    data_dir = './data/imagenet/tiny-imagenet-200/'
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
        'test': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }

    image_datasets = {x: t_datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss, n_correct, n_samples, pred, all_y = 0.0, 0, 0, [], []
    with torch.no_grad():
        for X, y in dataloaders['val']:
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

    valid_balanced_acc = round(balanced_accuracy_score(all_y, pred) * 100, 2)
    valid_loss = round(total_loss / n_samples, 2)
    valid_acc = round(n_correct / n_samples * 100, 2)
    print(
        # f"  train loss: {train_loss[epoch]:9.3f} |"
        # f"  train acc:  {train_acc[epoch]:9.3f}% |"
        f"  valid loss: {valid_loss:9.3f} |"
        f"  valid acc:  {valid_acc:9.3f}% |"
        f"  valid balanced acc:  {valid_balanced_acc:9.3f}% |"
    )

    # train(200, N_EPOCHS, lr, dataloaders['train'], dataloaders['val'], DEVICE, 'original', 0, mode)
