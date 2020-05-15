import matplotlib.image as mpimg
import torch.nn.functional as F
from matplotlib.pyplot import figure

from explainer import *
from helpfunctions import get_cross_val_acc


def evaluate(model, val_dl, k, explainers, image_class, path_root, subpath_healthy, subpath_diseased,
             subpath_classification, DEVICE, plot_diseased, plot_healthy, plot_classes):
    # get index for each class
    # actual: healthy prediction: healthy, true positive
    # actual: diseased prediction: healthy, false positive
    # actual: diseased prediction: diseased, true negative
    # actual: healthy prediction: diseased, false negative
    index_classes = [-1, -1, -1, -1]
    index_classes_image = [0, 0, 0, 0]
    # 6 images of plants with detected diseased TP
    index_diseased = []
    index_diseased_image = []
    # 6 images of healthy plants TN
    index_healthy = []
    index_healthy_image = []
    # Predict val dataset with final trained resnet, 1 = disease, 0 = no disease
    model.to(DEVICE)
    model.eval()
    pred, labels = [], []
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_ = model(X)
            _, y_label_ = torch.max(y_, 1)
            pred += y_label_.tolist()
            labels += y.tolist()
            ydata, preddata = y.tolist(), y_label_.tolist()
            # get images and labels for images which get displayed
            for i in range(0, len(y)):
                if plot_diseased and len(index_diseased) < k and preddata[i] == 1 and ydata[i] == 1:
                    index_diseased += [ydata[i]]
                    index_diseased_image.append(explain(model, X[i], ydata[i]))
                if plot_healthy and len(index_healthy) < k and preddata[i] == 0 and y[i] == 0:
                    index_healthy += [ydata[i]]
                    index_healthy_image.append(explain(model, X[i], ydata[i]))
                if plot_classes and index_classes[2] == -1 and preddata[i] == 1 and ydata[i] == 1:
                    index_classes[2] = ydata[i]
                    index_classes_image[2] = explain(model, X[i], ydata[i])
                if plot_classes and index_classes[3] == -1 and preddata[i] == 0 and ydata[i] == 1:
                    index_classes[3] = ydata[i]
                    index_classes_image[3] = explain(model, X[i], ydata[i])
                if plot_classes and index_classes[0] == -1 and preddata[i] == 0 and ydata[i] == 0:
                    index_classes[0] = ydata[i]
                    index_classes_image[0] = explain(model, X[i], ydata[i])
                if plot_classes and index_classes[1] == -1 and preddata[i] == 1 and ydata[i] == 0:
                    index_classes[1] = ydata[i]
                    index_classes_image[1] = explain(model, X[i], ydata[i])
    if not os.path.exists(path_root + subpath_healthy):
        os.makedirs(path_root + subpath_healthy)
    if not os.path.exists(path_root + subpath_diseased):
        os.makedirs(path_root + subpath_diseased)
    if not os.path.exists(path_root + subpath_classification):
        os.makedirs(path_root + subpath_classification)
    if plot_healthy:
        # save images of explainer in data
        # save healthy images which got detected
        for k in range(0, len(index_healthy)):
            for i in range(0, len(explainers)):
                index_healthy_image[k][i].savefig(
                    path_root + subpath_healthy + explainers[i] + str(k) + '.png',
                    bbox_inches='tight')

    if plot_diseased:
        # save images of explainer in data
        # save diseased images which got detected
        for k in range(0, len(index_diseased)):
            for i in range(0, len(explainers)):
                index_diseased_image[k][i].savefig(
                    path_root + subpath_diseased + explainers[i] + str(k) + '.png',
                    bbox_inches='tight')

    if plot_classes:
        # save images of every class to compare
        for k in range(0, len(image_class)):
            for i in range(0, len(explainers)):
                index_classes_image[k][i].savefig(
                    path_root + subpath_classification + explainers[i] + image_class[k] + '.png', bbox_inches='tight')


# evaluates the image according to the given model
# returns label, predicted label and the probity of it
def evaluate_id(image_id, ds, model, explainers, path_root, subpath, DEVICE):
    # evaluate predictions with model and create Images from explainers
    # # get first batch for evaluation
    # dataiter = iter(val_dl)
    # image1, label1 = next(dataiter)
    #
    # # print images of first batch
    # display_rgb_grid(torchvision.utils.make_grid(image1), 'images of batch 1')
    # # predict images of first batch
    # print('GroundTruth: ', ' '.join('%5s' % classes[label1[j]] for j in range(batch_size)))
    #
    # outputs1 = model(image1.to(DEVICE))
    #
    # _, predicted1 = torch.max(outputs1, 1)
    #
    # print('Predicted: ', ' '.join('%5s' % classes[predicted1[j]] for j in range(batch_size)))

    if not os.path.exists(path_root + subpath):
        os.makedirs(path_root + subpath)
    image, label = ds.get_by_id(image_id)
    if image is not None:
        model.to(DEVICE)
        image = torch.from_numpy(image).to(DEVICE)
        explained = explain(model, image, label)
        for i in range(0, len(explainers)):
            directory = path_root + subpath + explainers[i] + image_id + '.png'
            explained[i].savefig(directory, bbox_inches='tight')
        image = image[None]
        image = image.type('torch.FloatTensor').to(DEVICE)
        output = model(image)
        _, pred = torch.max(output, 1)
        prob = torch.max(F.softmax(output, 1)).item()
        return label, pred.item(), prob
    else:
        return -1, -1, -1


# compare given images of plantes in a plot
def plot_single_explainer(pathroot, subpath, explainers, image_names, title, roar):
    exnum = len(explainers)
    number_images = len(image_names)
    images = []
    for k in range(0, number_images):
        for i in range(0, exnum):
            try:
                images.append(mpimg.imread(pathroot + subpath + explainers[i] + image_names[k] + '.png'))
            except FileNotFoundError:
                print('image could not be loaded')
    number_images = len(images) // exnum
    fig = plt.figure(figsize=(6 * exnum + 2, 7 * number_images + 8))
    fig.suptitle(title, fontsize=35)
    for k in range(0, len(images) // exnum):
        for i in range(0, exnum):
            ax = fig.add_subplot(number_images, exnum, (i + 1) + k * exnum)
            plt.imshow(images[i + k * exnum])
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if k == 0:
                ax.set_title(explainers[i], fontsize=25)
            if i == 0:
                if subpath == 'classification/':
                    if image_names[k] == 'tp':
                        ax.set_ylabel('TP:\n Truth: healthy\n Prediction: healthy', fontsize=25)
                    if image_names[k] == 'fp':
                        ax.set_ylabel('FP:\n Truth: healthy\n Prediction: diseased', fontsize=25)
                    if image_names[k] == 'tn':
                        ax.set_ylabel('TN:\n Truth: diseased\n Prediction: diseased', fontsize=25)
                    if image_names[k] == 'fn':
                        ax.set_ylabel('FN:\n Truth: diseased\n Prediction: healthy', fontsize=25)
                else:
                    ax.set_ylabel('image ' + image_names[k], fontsize=25)

    fig.tight_layout()
    if not os.path.exists(pathroot + subpath + 'conclusion/'):
        os.makedirs(pathroot + subpath + 'conclusion/')
    fig.savefig(pathroot + subpath + 'conclusion/' + 'conclusion' + roar + '.png')
    plt.show()


def plot_explained_categories(model, val_dl, DEVICE, plot_diseased, plot_healthy, plot_classes, explainers, mode):
    path_root = './data' + mode + '/' + '/exp/'
    subpath_healthy = 'healthy/'
    subpath_diseased = 'diseased/'
    subpath_classification = 'classification/'
    image_class = ['tp', 'fp', 'tn', 'fn']
    number_images = 6
    image_indexed = [str(i) for i in range(1, number_images + 1)]
    # evaluate images and their classification
    evaluate(model, val_dl, number_images, explainers, image_class, path_root, subpath_healthy,
             subpath_diseased, subpath_classification, DEVICE, plot_diseased, plot_healthy, plot_classes)
    if plot_classes:
        plot_single_explainer(path_root, subpath_classification, explainers, image_class,
                              'Class comparison TP, FP, TN, FN on plant diseases', '')
    if plot_diseased:
        plot_single_explainer(path_root, subpath_diseased, explainers, image_indexed,
                              'comparison between detected diseased images', '')
    if plot_healthy:
        plot_single_explainer(path_root, subpath_healthy, explainers, image_indexed,
                              'comparison between detected healthy images', '')


def plot_explained_images(model, all_ds, DEVICE, explainers, image_ids, roar, mode):
    classes = ('healthy', 'diseased')
    path_root = './data/' + mode + '/' + 'exp/'
    subpath_single_image = 'single_image/'
    image_labels = np.zeros((len(image_ids), 4))
    image_pred = np.zeros((len(image_ids), 4))
    image_prob = np.zeros((len(image_ids), 4))
    number_images = 6
    image_indexed = []
    for i in range(1, number_images + 1):
        image_indexed.append(str(i))
    # evaluate for specific Image IDs
    for i, id in enumerate(image_ids):
        for k in range(1, 5):
            label, pred, prob = evaluate_id(str(k) + '_' + id, all_ds, model, explainers, path_root,
                                            subpath_single_image + id + '/', DEVICE)
            image_labels[i, k - 1] = label
            image_pred[i, k - 1] = pred
            image_prob[i, k - 1] = prob
    # plot created explainer
    for i, id in enumerate(image_ids):
        image_names = []
        for k in range(1, 5):
            image_names.append(str(k) + '_' + id)
        c1 = 'Truth for each day: '
        c2 = 'Prediction for each day: '
        prob = 'Probability of the prediction: '
        for k in range(0, 4):
            if image_pred[i, k] != -1:
                c1 = c1 + 'Day ' + str(k) + ': ' + classes[int(image_labels[i, k])] + ' '
                c2 = c2 + 'Day ' + str(k) + ': ' + classes[int(image_pred[i, k])] + ' '
                prob = prob + 'Day ' + str(k) + ': ' + str(round(image_prob[i, k] * 100, 2)) + ' '
        prediction = c1 + '\n' + c2 + '\n' + prob
        plot_single_explainer(path_root, subpath_single_image + id + '/', explainers, image_names,
                              'Plant comparison over days of ID: ' + id + ', roar method: ' + roar + '\n' + prediction,
                              roar)


# crossval acc for every removed percentage of each explainer
def plot_dev_acc(roar_values, roar_explainers, cv_iter, mode, model_type):
    # roar_explainers += ['random']
    colors = ['g', 'b', 'c', 'm', 'y', 'k', ]
    val = get_cross_val_acc('original', 0, cv_iter, mode, model_type)
    fig = figure(num=None, figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.plot([roar_values[0], 100], [val, val], 'r--', label='accuracy with 0% removed = ' + str(val) + '%')
    # plt.plot([roar_values[0], roar_values[-1]], [50, 50], 'k')
    for c, ex in enumerate(roar_explainers):
        acc_vals = []
        for roar_per in roar_values:
            acc = get_cross_val_acc(ex, roar_per, cv_iter, mode, model_type)
            print(acc)
            acc_vals.append(acc)
        if mode == 'plants':
            acc_vals.append(50)
        else:
            acc_vals.append(0.5)
        plt.plot(roar_values + [100], acc_vals, label=ex)
    min_acc = int(min(acc_vals) / 10) * 10
    max_acc = (1 + ((int(val)) / 10)) * 10
    plt.title(str(cv_iter) + ' cross val accuracy by increasing the removed image features of each saliency method')
    plt.xlabel('% of the image features removed from image')
    plt.ylabel('model accuracy')
    plt.axis([roar_values[0], 100, min_acc, max_acc])
    plt.legend(loc='lower left')
    plt.savefig('./data/' + mode + '/' + 'plots/accuracy_roar_comparison')
    plt.show()
    plt.close(fig)


def plot_single_image(model, id, ds, explainer, DEVICE, mode, set_title):
    image_normalized, label = ds.get_by_id(id)
    output = model(torch.unsqueeze(image_normalized, 0).to(DEVICE))
    _, pred = torch.max(output, 1)
    image, label = ds.get_original_by_id(id)
    classname = ds.get_class_by_label(label)
    pred_classname = ds.get_class_by_label(pred)
    title = explainer + ' heat-map on image ' + str(
        id) + '\nactual class: ' + classname + '\npredicted class: ' + pred_classname
    print(title + '\n######')
    fig, ax = plt.subplots()
    # org = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))
    # org_img_edged = preprocessing.scale(np.array(org, dtype=float)[:, :, 1] / 255)
    # org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
    # # Compute the Canny filter for two values of sigma
    # org_img_edged = feature.canny(org_img_edged, sigma=3)
    if explainer == 'Original':
        title = 'image ' + str(id) + '\nclass: ' + classname
        ax.set_title(title)
        org_im, _ = viz.visualize_image_attr(None,
                                             np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                             method="original_image", use_pyplot=False)
        plt.imshow(np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)))
    else:
        ax.set_title(title)
        explained = explain_single(model, image_normalized, label, explainer, True, DEVICE, mode)
        if explainer is not 'gradcam':
            explained = ndi.gaussian_filter(explained, 3)
        # ax.imshow(org_img_edged, cmap=plt.cm.binary)
        # ax.imshow(explained, cmap='viridis', vmin=np.min(explained), vmax=np.max(explained),
        #           alpha=0.4)

        viz.visualize_image_attr(np.expand_dims(explained, axis=2),
                                 np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)), sign="positive",
                                 method="blended_heat_map",
                                 show_colorbar=False, use_pyplot=False, plt_fig_axis=(fig, ax), cmap='viridis',
                                 alpha_overlay=0.6)
    ax.tick_params(axis='both', which='both', length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.grid(b=False)
    plt.show()
    if not os.path.exists('./data/' + mode + '/' + 'exp/saliency_single_image_eval/'):
        os.makedirs('./data/' + mode + '/' + 'exp/saliency_single_image_eval/')
    fig.savefig('./data/' + mode + '/' + 'exp/saliency_single_image_eval/' + str(
        id) + '_image_explained_with_' + explainer + '.png')
    plt.close('all')


def create_comparison_saliency(model_path, ids, ds, explainers, DEVICE, mode, model_type):
    title = 'comparison of saliency methods'
    len_ids = len(ids)
    len_explainer = len(explainers)
    w, h = 9 * len_explainer + 2, 10 * len_ids + 5

    fig = plt.figure(figsize=(w, h))
    decription = ''
    # fig.subplots_adjust(top=5, bottom=1, wspace=0.1, hspace=0.1)
    fig.suptitle(title, fontsize=80)

    for c_i, i in enumerate(ids):
        n_classes = 200
        if mode is 'plants':
            n_classes = 2
        model = get_model(DEVICE, n_classes, mode, model_type)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        image_normalized, label = ds.get_by_id(i)
        output = model(torch.unsqueeze(image_normalized, 0).to(DEVICE))
        _, pred = torch.max(output, 1)
        image, label = ds.get_original_by_id(i)
        classname = ds.get_class_by_label(label)
        pred_classname = ds.get_class_by_label(pred)
        image, label = ds.get_original_by_id(i)
        model.eval()
        # Edge detection of original input image
        org = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))
        org_img_edged = preprocessing.scale(np.array(org, dtype=float)[:, :, 1] / 255)
        org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
        # Compute the Canny filter for two values of sigma
        org_img_edged = feature.canny(org_img_edged, sigma=3)
        for c_ex, ex in enumerate(explainers):

            ax = fig.add_subplot(len_ids, len_explainer, (c_ex + 1) + c_i * len_explainer)

            if ex == 'Original':
                org_im, _ = viz.visualize_image_attr(None,
                                                     np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                                     method="original_image", use_pyplot=False)
                plt.imshow(np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)))
            else:
                explained = explain_single(model, image_normalized, label, ex, False, DEVICE, mode)
                if ex is not 'gradcam':
                    explained = ndi.gaussian_filter(explained, 3)
                # comment to use edged image
                # org_img_edged = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))

                viz.visualize_image_attr(np.expand_dims(explained, axis=2),
                                         org_img_edged,
                                         sign="positive", method="blended_heat_map",
                                         show_colorbar=False, use_pyplot=False, plt_fig_axis=(fig, ax), cmap='viridis',
                                         alpha_overlay=0.6)
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.grid(b=False)
            if c_i == 0:
                ax.set_title(ex, fontsize=40)
            if c_ex == 0:
                corect_pred = 'incorrectly'
                if classname == pred_classname:
                    corect_pred = 'correctly'
                ax.set_ylabel(
                    # 'image class ' + classname + '\nprediction: ' + pred_classname
                    'image class ' + str(label) + '\n' + corect_pred + ' classified'
                    , fontsize=40)
        del image
        decription += 'class ' + str(label) + ' = ' + classname + '\n'
    fig.text(0.02, 0, decription, fontsize=40)
    rect = (0, 0.08, 1, 0.95)
    fig.tight_layout(rect=rect, h_pad=8, w_pad=8)
    if not os.path.exists('./data/' + mode + '/' + 'exp/saliency_comparison/'):
        os.makedirs('./data/' + mode + '/' + 'exp/saliency_comparison/')
    fig.savefig('./data/' + mode + '/' + 'exp/saliency_comparison/comparison_of_saliency_methods.png')
    plt.close('all')
