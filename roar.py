from plots import *

# roar_explainers = ['guided_gradcam', 'random', 'gradcam', 'noisetunnel_gaussian',
#                    'guided_gradcam_gaussian', 'noisetunnel', 'Integrated_Gradients']
ids_imagenet = [x * 500 for x in range(8)]
# ids_imagenet = [x * 1 for x in range(8)]
ids_and_labels_imagenet = [(x * 500, x) for x in range(8)]
# ids_and_labels_imagenet = [(x * 1, x) for x in range(8)]
ids_roar = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
# ids_roar_exp = [0,2,3,5,7]
ids_roar_exp = [1, 3, 4, 6]
# ids_roar_exp = [1]
ids_and_labels_plants = [('3_Z18_4_1_1', 1), ('3_Z17_1_0_0', 1), ('3_Z16_2_1_1', 1), ('3_Z15_2_1_2', 1),
                         ('3_Z8_4_0_0', 1), ('3_Z8_4_1_2', 1), ('3_Z1_3_1_1', 0), ('3_Z2_1_0_2', 0)]
root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
subpath_heatmaps = 'heatmaps/heatmaps'
subpath = 'roar/'
n_classes_plants = 2
n_classes_imagenet = 200


# applying the explainers to an roar trained image
# interpretation/explaination of modified roar Images
# Axes: removed % of image features and explainers
def roar_comparison_explained(mode, DEVICE, explainers, roar_expl_im_values, model_type):
    # explainers = ['noisetunnel', 'gradcam', 'guided_gradcam', 'noisetunnel_gaussian', 'guided_gradcam_gaussian']
    # roar_expl_im_values = [0, 10, 20, 30, 50, 70, 90, 100]
    path_exp = './data/' + mode + '/' + 'exp/'
    trained_roar_models = './data/' + mode + '/'  'models/' + model_type + 'trained_model_roar'
    roar_expl_im_values = [0] + roar_expl_im_values
    if mode == 'imagenet':
        n_classes = n_classes_imagenet
        ids_and_labels = ids_and_labels_imagenet

    else:
        n_classes = n_classes_plants
        ids_and_labels = ids_and_labels_plants
    font = {
        'size': 15,
        # 'family': 'serif',
        # 'serif': ['Computer Modern']
        'family': 'sans-serif',
        'serif': ['Computer Modern Sans serif']
    }

    rc('font', **font)
    w, h = 8 * len(explainers), 7 * len(roar_expl_im_values) + 3
    for k in ids_roar_exp:
        if mode == 'plants':
            id = str(3) + '_' + ids_roar[k]
        else:
            id = ids_imagenet[k]
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax.set_ylabel('% Removed', size=60)
        fig.suptitle(
            # "modified image " + str(id)  + " according to ROAR framework with applied interpretation of its saliency method",
            'Visual explanation of a modified image',
            size=80)
        fig.subplots_adjust(top=0.92)
        print('plotting modified image:' + str(id) + ' according to roar')
        all_ds = Spectralloader([ids_and_labels[k]], root, mode, 'specific')
        image, label = all_ds.get_original_by_id(id)
        for c_e, a in enumerate(explainers):
            ax = fig.add_subplot(len(roar_expl_im_values) + 1, len(explainers),
                                 c_e + 1)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(a, fontsize=40)
            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if c_e == 0:
                ax.set_ylabel('original image', fontsize=40)
        for c_ex, ex in enumerate(explainers):
            # loading heatmap of corresponding explainer
            if mode == 'plants':
                with open(path_exp + subpath_heatmaps + '_' + model_type + '_' + ex + '.pkl', 'rb') as f:
                    mask = pickle.load(f)
            else:
                # imagenet mask gets loaded in Dataset
                mask = None
            print('applying ' + ex + ' to image')
            for c_r, i in enumerate(roar_expl_im_values):
                # select 3 day image of image ID
                # loading model of explainer for corresponding remove value
                all_ds_copy = deepcopy(all_ds)
                if i == 0:
                    model = get_model(DEVICE, n_classes, mode, model_type)
                    original_trained_model = './data/' + mode + '/' + 'models/VGGtrained_model_original.pt'
                    model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
                    model.eval()

                else:
                    model = get_model(DEVICE, n_classes, mode, model_type)
                    model.load_state_dict(
                        torch.load(trained_roar_models + '_' + ex + '_' + str(i) + '.pt', map_location=DEVICE))
                    model.eval()
                    all_ds_copy.apply_roar_single_image(i, mask, id, 'mean', ex)
                # plot_explained_images(model, all_ds, DEVICE, explainers, image_ids, str(i) + "%removed")
                image, label = all_ds_copy.get_original_by_id(id)
                model = model.to(DEVICE)
                image = image.to(DEVICE)
                ax = fig.add_subplot(len(roar_expl_im_values) + 1, len(explainers),
                                     (c_ex + 1) + (c_r + 1) * len(explainers))
                explained = explain_single(model, image, label, ex, True, DEVICE, mode)
                org_img = np.transpose(image.squeeze().cpu().detach().numpy(), to_RGB)

                if ex is not 'gradcam':
                    explained = ndi.gaussian_filter(explained, 3)

                if mode == 'imagenet':
                    explained = np.expand_dims(explained, axis=2)
                    viz.visualize_image_attr(explained,
                                             org_img,
                                             sign="positive", method="blended_heat_map",
                                             show_colorbar=False, use_pyplot=False, plt_fig_axis=(fig, ax),
                                             cmap='viridis',
                                             alpha_overlay=0.6)
                else:
                    # Edge detection of original input image
                    org_img_edged = preprocessing.scale(np.array(org_img, dtype=float)[:, :, 1] / 255)
                    org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
                    # Compute the Canny filter for two values of sigma
                    org_img_edged = feature.canny(org_img_edged, sigma=3)
                    ax.imshow(org_img_edged, cmap=plt.cm.binary)
                    ax.imshow(explained, cmap='viridis',
                              # vmin=np.min(explained), vmax=np.max(explained),
                              alpha=0.4)
                ax.tick_params(axis='both', which='both', length=0)
                if c_ex == 0:
                    ax.set_ylabel(str(i) + '%', size=40)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.grid(b=False)
        rect = (0, 0.08, 1, 0.95)
        fig.tight_layout(rect=rect, h_pad=4, w_pad=4)
        fig.savefig(path_exp + subpath + model_type + '/comparison_explained_roar_image_' + str(id) + '.png')
        fig.clear()


# plotting the roar trained images
# comparison of modified roar Images
# Axes: removed % of image features and explainers
def roar_comparison(mode, roar_explainers, cv_iter, roar_values, model_type):
    # roar_explainers = ['random'] + roar_explainers
    # roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    path_exp = './data/' + mode + '/' + 'exp/'
    subpath = 'roar/' + model_type + '/'
    if mode == 'plants':
        image_ids = ids_roar
        ids_and_image_labels = ids_and_labels_plants
    else:
        image_ids = ids_imagenet
        ids_and_image_labels = ids_and_labels_imagenet
    print('plotting images with removed values')
    font = {
        'size': 15,
        # 'family': 'serif',
        # 'serif': ['Computer Modern']
        'family': 'sans-serif',
        'serif': ['Computer Modern Sans serif']
    }

    rc('font', **font)
    w, h = 7 * len(roar_explainers), 7 * len(roar_values) + 5
    for k in ids_roar_exp:
        all_ds = Spectralloader([ids_and_image_labels[k]], root, mode, 'specific')

        fig = plt.figure(figsize=(w, h))
        fig.suptitle(
            'ROAR: RemOve And Retrain',
            # "image " + str(image_ids[k]) + " modificed according to ROAR framework",
            size=80)
        # ax = fig.add_subplot(111)
        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['right'].set_color('none')
        # ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        #
        # ax.set_ylabel('% Removed', size=60)
        fig.subplots_adjust(top=0.92)

        print('modifing image: ' + str(image_ids[k]))
        if not os.path.exists(path_exp + subpath):
            os.makedirs(path_exp + subpath)
        for c_ex, ex in enumerate(roar_explainers):
            # loading heatmap of corresponding explainer
            if mode == 'plants':
                with open(path_exp + subpath_heatmaps + '_' + model_type + '_' + ex + '.pkl', 'rb') as f:
                    mask = pickle.load(f)
            else:
                # imagenet mask gets loaded in Dataset
                mask = None
            print('appling ' + ex + ' to image')
            for c_r, roar_per in enumerate(roar_values):
                if mode == 'plants':
                    id = str(3) + '_' + str(image_ids[k])
                else:
                    id = image_ids[k]
                all_ds_tmp = deepcopy(all_ds)
                sub_path = str(roar_per) + '%_of_' + ex + '.sav'
                path = './data/' + mode + '/' + 'plots/values/' + sub_path
                if roar_per == 0:
                    acc, _ = get_cross_val_acc('original', roar_per, cv_iter, mode, model_type)
                else:
                    all_ds_tmp.apply_roar_single_image(roar_per, mask, id, 'comp', ex)
                    acc, _ = get_cross_val_acc(ex, roar_per, cv_iter, mode, model_type)
                image, label = all_ds_tmp.get_original_by_id(id)
                # show_image(image, 'modified image')
                # create ROAR plot
                ax = fig.add_subplot(len(roar_values), len(roar_explainers),
                                     (c_ex + 1) + c_r * len(roar_explainers))
                ax.tick_params(axis='both', which='both', length=0)
                if c_ex == 0:
                    ax.set_ylabel(str(roar_per) + '%', fontsize=40)
                if c_r == 0:
                    ax.set_title(ex + '\n' + str(round(acc, 2)) + '%', fontsize=40)
                else:
                    ax.set_title(str(round(acc, 2)) + '%', fontsize=40)
                ax.imshow(np.transpose(image, (1, 2, 0)))
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
            plt.grid(b=False)
        rect = (0, 0.08, 1, 0.95)
        fig.tight_layout(rect=rect, h_pad=5, w_pad=5)
        fig.savefig(path_exp + subpath + 'comparison_roar_images' + str(id) + '.png')
        fig.clear()
