import sys

from explainer import *
from helpfunctions import *
from plots import *
from roar import *
from spectralloader import *


def main():
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # plant or imagenet DS
    modes = ['plants', 'imagenet']
    mode = modes[1]

    # train and modify dataset
    # resizes all images and replaces them in folder
    resize_imagenet = False
    retrain = False

    # explain image and create comparison
    # only available for plant
    plot_classes, plot_categories = False, False
    # comparison for image ID
    plot_for_image_id = False
    # expain images seperate
    explain_images_single = True

    # ROAR
    roar_create_mask = False
    roar_train = False
    plot_roar_curve = False
    roar_comp = False
    roar_expl_comp = False

    # CNN default learning parameters
    # dafault training Values for plant dataset, resnet18 with lr = 0.00015, Epochs = 120, batchsize = 20
    N_EPOCHS = 120
    lr = 0.00015
    n_classes = 2
    batch_size = 20
    cv_iterations_total = 5
    test_size = 482


    image_ids = ['Z18_4_1_1', 'Z17_1_0_0', 'Z16_2_1_1', 'Z15_2_1_2', 'Z8_4_0_0', 'Z8_4_1_2', 'Z1_3_1_1', 'Z2_1_0_2']
    # explainer for orignal explaination
    explainers = ['Original', 'saliency', 'Integrated_Gradients',
                  'noisetunnel',
                  'guided_gradcam', 'gradcam', 'LRP',
                  # 'Noise Tunnel stev 2'
                  ]
    explainers = ['gradcam']

    # ROAR explainer to be applied
    roar_explainers = ['gradcam', 'guided_gradcam', 'guided_gradcam_gaussian',
                       'noisetunnel', 'noisetunnel_gaussian', 'Integrated_Gradients']
    roar_explainers = ['gradcam', 'guided_gradcam', 'LRP',
                       'noisetunnel',
                       'random', 'Integrated_Gradients']
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    roar_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    roar_values = [10, 30, 70, 90]
    cv_it_to_calc = [0]
    if mode == 'imagenet':
        if resize_imagenet:
            val_format()
            upscale_imagenet()
        n_classes = 200
        N_EPOCHS = 15
        lr = 0.001
        batch_size = 50
        # print('nr epochs: ' + str(N_EPOCHS))
        # print('batch_size ' + str(batch_size))
        # print('lr ' + str(lr))
        cv_iterations_total = 1
        test_size = 10000
        image_ids = [x * 500 for x in range(5)]

    train_labels, valid_labels, all_data, labels = load_labels(mode)
    sss = StratifiedShuffleSplit(n_splits=cv_iterations_total, test_size=test_size, random_state=0)
    # save the explainer images of the figures
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    if not os.path.exists('./data/' + mode + '/' + 'models/'):
        os.makedirs('./data/' + mode + '/' + 'models/')
    if not os.path.exists('./data/' + mode + '/' + 'plots/'):
        os.makedirs('./data/' + mode + '/' + 'plots/')
    if not os.path.exists('./data/' + mode + '/' + 'plots/values/'):
        os.makedirs('./data/' + mode + '/' + 'plots/values/')

    trained_roar_models = './data/' + mode + '/' + 'models/trained_model_roar'
    original_trained_model = './data/' + mode + '/' + 'models/trained_model_original.pt'
    # original_trained_model = trained_roar_models + '_gradcam_10.pt'
    root = '/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/'
    path_exp = './data/' + mode + '/' + 'exp/'


    # use to evaluate model
    # original_model = get_model(DEVICE, n_classes, mode)
    # original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
    # train_imagenet(original_model,N_EPOCHS,lr,batch_size,DEVICE,mode)

    # train model or use trained model from last execution
    if retrain:
        train_cross_val(sss, all_data, labels, root, mode, batch_size, n_classes, N_EPOCHS, lr, DEVICE,
                        original_trained_model, cv_it_to_calc)

    # only plants
    # create comparison for explained plants images -> TP,FP,TN,FN comparison; class comparison healthy/diseased
    if plot_classes or plot_categories:
        original_model = get_model(DEVICE, n_classes, mode)
        original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
        print('loading validation dataset')
        val_ds = Spectralloader(valid_labels, root, mode, 'val')
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, )
        # evaluate images and their classification
        print('creating explainer plots for specific classes')
        plot_explained_categories(original_model, val_dl, DEVICE, plot_categories, plot_categories, plot_classes,
                                  explainers, mode)

    # plot comparison of explained Images for specified IDs and explainers
    if plot_for_image_id:
        print('loading whole dataset')
        all_ds = Spectralloader(all_data, root, mode, 'all')
        original_model = get_model(DEVICE, n_classes, mode)
        original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
        print('creating explainer plots for specified images')
        if mode == 'plants':
            plot_explained_images(original_model, all_ds, DEVICE, explainers, image_ids, 'original', mode)
        else:
            create_comparison_saliency(original_trained_model, image_ids, all_ds, explainers, DEVICE, mode)

    # create a mask for specified IDS and explainers for given range
    # containing one heatmap per specified specified images
    if roar_create_mask:
        all_ds = Spectralloader(all_data, root, mode, 'all')
        original_model = get_model(DEVICE, n_classes, mode)
        original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
        input_cmd = sys.argv
        # whole mask start:0 end 105000
        mask_range_start = 0
        mask_range_end = 105000
        # use cmd input variables if available
        if len(input_cmd) > 2:
            mask_range_start = int(input_cmd[1])
            mask_range_end = int(input_cmd[2])
        print('start: ' + str(mask_range_start) + ' end: ' + str(mask_range_end))
        print('creating for ROAR mask')
        create_mask(original_model, all_ds, path_exp, DEVICE, roar_explainers, mode,
                    mask_range_start, mask_range_end,
                    replace_existing=True)
        print('mask for ROAR created')

    # ROAR remove and retrain applied to all specified explainers and remove percentages
    if roar_train:
        """ roar curve

        Args:
            roar_explainers (list): List of saliency methods to be evaluated
            roar_values (list): List of percentages to be removed from the image
            DEVICE (string): device to run on
            mode (string): mode imagenet or plants
            cv_it_to_calc (number): number of crossval iterations to be trained
        """
        train_roar_ds(path_exp, roar_values, trained_roar_models, all_data, labels, batch_size,
                      n_classes, N_EPOCHS, lr, DEVICE, roar_explainers, sss, root, mode, cv_it_to_calc)

    # plot the acc curves of all trained ROAR models
    if plot_roar_curve:
        """ roar curve

       Args:
           roar_explainers (list): List of saliency methods to be evaluated
           roar_values (list): List of percentages to be removed from the image
           mode (string): mode imagenet or plants
           cv_iterations_total (number): number of crossval iterations -> must be trained before
       """
        plot_dev_acc(roar_values, roar_explainers, cv_iterations_total, mode)

    # comparison of modified roar Images
    if roar_comp:
        """ roar comparison

        Args:
            roar_explainers (list): List of saliency methods to be evaluated
            roar_values (list): List of percentages to be removed from the image
            mode (string): mode imagenet or plants
            cv_iterations_total (number): number of crossval iterations -> must be trained before
        """
        print('creating ROAR comparison plot')
        roar_comparison(mode, roar_explainers, cv_iterations_total, roar_values)

    # interpretation/explaination of modified roar Images
    if roar_expl_comp:
        """ roar comparison explained

        Args:
            roar_explainers (list): List of saliency methods to be evaluated
            roar_values (list): List of percentages to be removed from the image
            DEVICE (string): device to run on
            mode (string): mode imagenet or plants
        """
        print('creating ROAR explanation plot')
        roar_comparison_explained(mode, DEVICE, roar_explainers, roar_values)

    # create single plots to explain a image
    if explain_images_single:
        """ explain images

        Args:
            explainers (list): List of saliency methods to be evaluated
            image_ids (list): List of IDs to be evaluated
            DEVICE (string): device to run on
            mode (string): mode imagenet or plants
        """
        all_ds = Spectralloader(all_data, root, mode, 'all')
        original_model = get_model(DEVICE, n_classes, mode)
        original_model.load_state_dict(torch.load(original_trained_model, map_location=DEVICE))
        for ex in explainers:
            for i in image_ids:
                plot_single_image(original_model, i, all_ds, ex, DEVICE, mode)

if __name__ == '__main__':
    main()
