import os
import pickle

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

import numpy as np

from torchvision.transforms import transforms

# from learning to RGB
to_RGB = (1, 2, 0)
# from RGB to learning
to_learning = (2, 0, 1)


#
# transpose image to display learned images
def display_rgb_grid(img, title):
    print(title + ' image displayed with shape ' + str(img.shape))
    plt.title(title)
    npimg = img.numpy()
    # plt.imshow(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# display rgb image
# parameter img is tensor
def display_rgb(img, title, path, name):
    plt.title(title)
    # img_tr = np.transpose(img.squeeze().cpu().detach().numpy(), to_RGB)

    trans = transforms.ToPILImage()
    im = trans(img)
    plt.imshow(im)
    plt.show()
    if not os.path.exists(path):
        os.makedirs(path)
    im.save(path + name)


def show_image(img, title):
    plt.title(title)
    img = np.transpose(img, to_RGB)
    plt.imshow(img)
    plt.show()
    # im = Image.fromarray((img_tr * 255).astype(np.uint8))


def to_rgb(image):
    return np.transpose(image, to_RGB)


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


def figure_to_image(fig):
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.text(0.0, 0.0, "Test", fontsize=45)
    ax.axis('off')

    canvas.draw()  # draw the canvas, cache the renderer

    return np.fromstring(canvas.tostring_rgb(), dtype='uint8')


def get_cross_val_acc(ex, roar_per, cv_iter, mode, model_type):
    print('roar acc with ' + str(roar_per) + '% image features removed of ' + ex)
    acc_list = []
    j = 0
    acc = 0
    missing_val = False
    try:
        if ex == 'original':
            for j in cv_iter:
                sub_path = ex + '_cv_it_' + str(j) + '.sav'
                path = './data/' + mode + '/' + 'plots/values/' + model_type + '/' + sub_path
                cur_acc = pickle.load(open(path, 'rb'))
                acc += cur_acc
                acc_list.append(cur_acc)
        else:
            for j in cv_iter:
                sub_path = str(roar_per) + '%_of_' + ex + '_cv_it_' + str(j) + '.sav'
                sub_path_2 = str(roar_per) + '%25_of_' + ex + '_cv_it_' + str(j) + '.sav'
                path_s = './data/' + mode + '/' + 'plots/values/' + model_type + '/'
                path_fin = path_s + sub_path
                if not os.path.isfile(path_fin):
                    path_fin = path_s + sub_path_2
                if os.path.isfile(path_fin):
                    cur_acc = pickle.load(open(path_fin, 'rb'))
                    acc += cur_acc
                    acc_list.append(cur_acc)
                else:
                    acc_list.append(None)
                    missing_val = True
        print(acc_list)
        out = round(acc / len(cv_iter), 2)
        if missing_val:
            out = -100
        return out, acc_list
    except ValueError:
        print(
            'accuracies for:' + ex + 'with ' + roar_per + ' removed image features in cross val iteration: ' + str(
                j) + ' are not yet evaluated')
