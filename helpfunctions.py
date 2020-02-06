import os
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
    plt.imshow(np.transpose(img, (1, 2, 0)))
    im = Image.fromarray(img)
    plt.show()
    if not os.path.exists(path):
        os.makedirs(path)
    im.save(path + name)

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