import os
import pickle

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# transpose image to display learned images
def display_rgb_grid(img, title):
    print(title + ' image displayed with shape ' + str(img.shape))
    plt.title(title)
    npimg = img.numpy()
    # plt.imshow(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# display rgb image
def display_rgb(img, title, path, name):
    plt.title(title)
    img_tr = np.transpose(img, (1, 2, 0))
    plt.imshow(img_tr)
    plt.show()
    im = Image.fromarray((img_tr * 255).astype(np.uint8))
    if not os.path.exists(path):
        os.makedirs(path)
    im.save(path + name)


def show_image(img, title):
    plt.title(title)
    img_tr = np.transpose(img, (1, 2, 0))
    plt.imshow(img_tr)
    plt.show()
    # im = Image.fromarray((img_tr * 255).astype(np.uint8))


def to_rgb(image):
    return np.transpose(image, (1, 2, 0))


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


def get_cross_val_acc(ex, roar_per, cv_iter, mode):
    j = 0
    acc = 0
    try:
        if ex == 'original':
            for j in range(cv_iter):
                sub_path = ex + '_cv_it_' + str(j + 1) + '.sav'
                path = './data/' + mode + '/' + 'plots/values/' + sub_path
                acc += pickle.load(open(path, 'rb'))
        else:
            for j in range(cv_iter):
                sub_path = str(roar_per) + '%_of_' + ex + '_cv_it_' + str(j) + '.sav'
                path = './data/' + mode + '/' + 'plots/values/' + sub_path
                acc += pickle.load(open(path, 'rb'))

        return acc / cv_iter
    except ValueError:
        print(
            'accuracies for:' + ex + 'with ' + roar_per + ' removed image features in cross val iteration: ' + str(
                j) + ' are not yet evaluated')
