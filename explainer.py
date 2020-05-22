from PIL import Image as PImage
from captum.attr import GuidedGradCam
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import Saliency
from captum.attr import visualization as viz
from captum.attr._core.guided_grad_cam import LayerGradCam
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import feature
from sklearn import preprocessing
from torchvision.models import VGG
from torch.autograd._functions import Resize

from cnn import *


# create single explainer of the image for the specified explainer


def explain_single(model, image, ori_label, explainer, bounded, DEVICE, mode):
    image = image.to(DEVICE)
    model = model.to(DEVICE)
    image.requires_grad = True
    input = image.unsqueeze(0)
    # input.requires_grad = True
    model.eval()
    c, h, w = image.shape
    heat_map = np.random.rand(h, w)
    image_mod = image[None]
    output = model(image_mod)
    _, pred = torch.max(output, 1)
    label = pred.item()
    if isinstance(model, VGG):
        last_layer = model.features[-3]
    else:
        last_layer = model.layer4

    def cut_and_shape(data):
        # c, h, w = data.shape
        # consider only the positive values
        for i in range(h):
            for k in range(w):
                for j in range(c):
                    if data[i][k][j] < 0:
                        data[i][k][j] = 0
        # reshape to 2D hxw
        d_img = data[:, :, 0] + data[:, :, 1] + data[:, :, 2]
        # d_img = data
        return d_img

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=label, **kwargs)
        return tensor_attributions

    if explainer == 'gradcam':
        # GradCam
        gco = LayerGradCam(model, last_layer)
        attr_gco = attribute_image_features(gco, input)
        att = attr_gco.squeeze(0).squeeze(0).cpu().detach().numpy()
        h_a, w_a = att.shape
        # for i in range(h_a):
        #     for k in range(w_a):
        #         if att[i][k] < 0:
        #             att[i][k] = 0
        gradcam = PImage.fromarray(att).resize((w, h), PImage.ANTIALIAS)
        heat_map = np.asarray(gradcam)


    elif explainer == 'guided_gradcam':
        gc = GuidedGradCam(model, last_layer)
        attr_gc = attribute_image_features(gc, input)
        heat_map = cut_and_shape(np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
        if bounded:
            heat_map = cut_top_per(heat_map)

    elif explainer == 'guided_gradcam_gaussian':
        gc = GuidedGradCam(model, last_layer)
        attr_gc = attribute_image_features(gc, input)
        heat_map = cut_and_shape(np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
        heat_map = ndi.gaussian_filter(heat_map, 7)

    elif explainer == 'noisetunnel':
        # IntegratedGradients Noise Tunnel
        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)
        attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0,
                                              nt_type='smoothgrad_sq',
                                              n_samples=2,
                                              # stdevs=0.2
                                              )
        heat_map = cut_and_shape(np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
        if bounded:
            heat_map = cut_top_per(heat_map)

    elif explainer == 'noisetunnel_gaussian':
        # IntegratedGradients Noise Tunnel
        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)
        attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                              # n_samples=5,
                                              n_samples=2,
                                              # stdevs=0.2
                                              )
        heat_map = cut_and_shape(np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
        heat_map = ndi.gaussian_filter(heat_map, 7)

    elif explainer == 'Integrated_Gradients':
        # IntegratedGradients
        ig = IntegratedGradients(model)
        attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
        heat_map = cut_and_shape(np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)))
        if bounded:
            heat_map = cut_top_per(heat_map)

    elif explainer == 'saliency':
        saliency = Saliency(model)
        grads = saliency.attribute(input, target=label)
        heat_map = cut_and_shape(np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0)))
        if bounded:
            heat_map = cut_top_per(heat_map)

    elif explainer == 'LRP':
        # model.to('cuda:0')
        # summary(model, (3, 255, 213), batch_size=20)
        # model.to(DEVICE)

        # CAPTUM lrp
        # lrp = LRP(model)
        # attr_lrp, delta = attribute_image_features(lrp, input, return_convergence_delta=True)
        # heat_map = cut_and_shape(np.transpose(attr_lrp.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # lrp FROM LOCAL LIB
        # print(model)
        # print("Layerwise_Relevance_Propagation")
        # train_imagenet(model,None,None,50,DEVICE,'imagenet')
        # import from local lib
        # investigator.py del layers remove for VGG only necessary for ResNet
        import innvestigator
        import settings as set
        original_trained_model = './data/imagenet/models/trained_model_original.pt'
        data_LRP_stored = './data/imagenet/exp/lrp'
        set.settings["model_path"] = original_trained_model
        set.settings["data_path"] = data_LRP_stored
        set.settings["ADNI_DIR"] = ''
        set.settings["train_h5"] = ''
        set.settings["val_h5"] = ''
        set.settings["holdout_h5"] = ''
        # # Convert to innvestigate model
        # show_image(input.squeeze().cpu().detach().numpy(), 'test')

        inn_model = innvestigator.InnvestigateModel(model, lrp_exponent=2,
                                                    method="e-rule",
                                                    beta=.5,
                                                    DEVICE=DEVICE)
        model_prediction, heat_map = inn_model.innvestigate(in_tensor=input)

        heat_map = cut_and_shape(np.transpose(heat_map.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
        if bounded:
            heat_map = cut_top_per(heat_map)

    torch.cuda.empty_cache()
    # assert (heat_map.shape == torch.Size([h, w])), "heatmap shape: " + str(
    #     heat_map.shape) + " does not match image shape: " + str(torch.Size([h, w]))
    if mode == 'plants':
        heat_map = PImage.fromarray(heat_map).resize((255, 213), PImage.ANTIALIAS)
        heat_map = np.asarray(heat_map)
        # heat_map = Resize.apply(heat_map, (255, 213))
    return heat_map


# create a mask with all heat_maps for specified dataset
def create_mask(model, model_name, dataset, path, DEVICE, roar_explainers, mode, range_start, range_end,
                replace_existing):
    if mode == 'imagenet':
        create_mask_imagenet(model, dataset, path, DEVICE, roar_explainers, replace_existing, range_start, range_end)
    else:
        d_length = dataset.__len__()
        model.to(DEVICE)
        heat_maps = {}
        for ex in roar_explainers:
            heat_maps[ex] = {}
        text = 'creating heatmaps for '
        for i in roar_explainers:
            text = text + i + ' '
        with tqdm(total=len(roar_explainers) * d_length, desc=text, ncols=100 + len(roar_explainers) * 15) as progress:
            for i in range(0, d_length):
                image, label = dataset.__getitem__(i)
                image = torch.Tensor(image).to(DEVICE)
                for ex in roar_explainers:
                    progress.update(1)
                    heat_maps[ex][str(dataset.get_id_by_index(i))] = explain_single(model, image, label, ex, False,
                                                                                    DEVICE, mode)
            if not os.path.exists(path + 'heatmaps'):
                os.makedirs(path + 'heatmaps')
            for ex in roar_explainers:
                pickle.dump(heat_maps[ex], open(path + 'heatmaps/heatmaps_' + model_name + '_' + ex + '.pkl', 'wb'))
                print('heatmap saved')


# create a mask with all heat_maps for specified dataset
def create_mask_imagenet(model, dataset, path, DEVICE, roar_explainers, replace_existing, range_start, range_end):
    d_length = dataset.__len__()
    model.to(DEVICE)
    heat_maps = {}

    for ex in roar_explainers:
        if not os.path.exists(path + '/heatmaps/' + ex):
            os.makedirs(path + '/heatmaps/' + ex)
        heat_maps[ex] = {}
    text = 'creating heatmaps for '
    for i in roar_explainers:
        text = text + i + ' '
    with tqdm(total=len(roar_explainers) * (range_end - (range_start - 1)), desc=text,
              ncols=100 + len(roar_explainers) * 15) as progress:
        for i in range(range_start, range_end + 1):
            id = dataset.get_id_by_index(i)
            image, label = dataset.__getitem__(id)
            image = image.to(DEVICE)
            for ex in roar_explainers:
                path_item = path + '/heatmaps/' + ex + '/' + str(id) + '.pkl'
                if (not os.path.isfile(path_item)) or (replace_existing):
                    tmp = explain_single(model, image, label, ex, False, DEVICE, 'imagenet')
                    pickle.dump(tmp, open(path_item, 'wb'))
                progress.update(1)


# cut top x Percentage of data and clips it to max
def cut_top_per(data):
    h, w = data.shape
    percentile = np.percentile(data, 98.5)
    # consider only the positive values
    for i in range(h):
        for k in range(w):
            if data[i][k] > percentile:
                data[i][k] = percentile
    # reshape to 2D hxw
    d_img = data
    return d_img


# create all explainers for a given image
# more preformat
def explain(model, image, label):
    print("creating images")
    input = image.unsqueeze(0)
    input.requires_grad = True
    model.eval()
    c, h, w = image.shape
    # Edge detection of original input image
    org = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))
    org_img_edged = preprocessing.scale(np.array(org, dtype=float)[:, :, 1] / 255)
    org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
    # Compute the Canny filter for two values of sigma
    org_img_edged = feature.canny(org_img_edged, sigma=3)

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=label, **kwargs)
        return tensor_attributions

    def detect_edge(activation_map):
        # org = np.zeros((h, w), dtype=float) + org_img_edged
        # org = np.asarray(org_img_edged)[:, :, np.newaxis]
        fig, ax = plt.subplots()
        ax.imshow(org_img_edged, cmap=plt.cm.binary)
        ax.imshow(activation_map, cmap='viridis', vmin=np.min(activation_map), vmax=np.max(activation_map),
                  alpha=0.4)
        ax.tick_params(axis='both', which='both', length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.close('all')
        return fig

    # normalize and reshape
    def normalize(data):
        # consider only the positive values
        for i in range(h):
            for k in range(w):
                for j in range(c):
                    # if label == 1:
                    if data[i][k][j] < 0:
                        data[i][k][j] = 0

        # reshape to hxw
        d_img = data[:, :, 0] + data[:, :, 1] + data[:, :, 2]
        max = np.max(d_img)
        mean = np.mean(d_img)
        min = 0
        designated_mean = 0.25
        factor = (designated_mean * max) / mean
        # normalize
        for i in range(h):
            for k in range(w):
                d_img[i][k] = (d_img[i][k] - min) / (max - min)
                if d_img[i][k] > 1:
                    d_img[i][k] = 1
        # apply gaussian
        d_img = ndi.gaussian_filter(d_img, 7)

        return d_img

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    # saliency
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=label)
    grads = normalize(np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0)))

    # IntegratedGradients
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = normalize(np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)))

    # IntegratedGradients Noise Tunnel
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=2,
                                          # stdevs=0.2
                                          )

    attr_ig_nt = normalize(np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))

    # IntegratedGradients Noise Tunnel
    attr_ig_nt2 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                           # n_samples=5,
                                           n_samples=2,
                                           stdevs=2.0
                                           )

    attr_ig_nt2 = normalize(np.transpose(attr_ig_nt2.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))

    # GuidedGradCam
    gc = GuidedGradCam(model, model.layer4)
    attr_gc = attribute_image_features(gc, input)
    attr_gc = normalize(np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))

    # GradCam Original Layer 4
    gco = LayerGradCam(model, model.layer4)
    attr_gco = attribute_image_features(gco, input)

    # GradCam
    att = attr_gco.squeeze(0).squeeze(0).cpu().detach().numpy()
    # gco_int = (att * 255).astype(np.uint8)
    gradcam = PImage.fromarray(att).resize((w, h), PImage.ANTIALIAS)
    np_gradcam = np.asarray(gradcam)

    f2 = detect_edge(grads)
    f3 = detect_edge(attr_ig)
    f4 = detect_edge(attr_ig_nt)
    f6 = detect_edge(attr_gc)
    f7 = detect_edge(np_gradcam)
    f8 = detect_edge(attr_ig_nt2)
    f1, a1 = viz.visualize_image_attr(None, np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      method="original_image", use_pyplot=False)
    return [f1, f2, f3, f4, f6, f7, f8]

# ----------------------------------------------------------------
# code to display images with the captum lib

# original_image = detect_edge()
# # Original Image
# # Overlayed Gradient Magnitudes saliency
# f2, a2 = viz.visualize_image_attr(grads, original_image, sign="positive", method="blended_heat_map", use_pyplot=False)
# # Overlayed Integrated Gradients
# f3, a3 = viz.visualize_image_attr(attr_ig, original_image, sign="positive", method="blended_heat_map", use_pyplot=False)
# # Overlayed Noise Tunnel
# f4, a4 = viz.visualize_image_attr(attr_ig_nt, original_image, sign="positive",method="blended_heat_map", use_pyplot=False)
#
# # # DeepLift
# # f5, a5 = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all",
# #                                   # show_colorbar=True, title="Overlayed DeepLift"
# #                                   )
# # f5 = detect_edge(attr_dl)
#
# # GuidedGradCam
# f6, a6 = viz.visualize_image_attr(attr_gc, original_image, sign="positive", method="blended_heat_map", show_colorbar=False, use_pyplot=False)
#
# # GradCam
# f7, a7 = viz.visualize_image_attr(np_gradcam, original_image, sign="positive", method="blended_heat_map", show_colorbar=False, use_pyplot=False)
#
# # GradCam original image
# f8, a8 = viz.visualize_image_attr(gradcam_orig, original_image, sign="absolute_value", method="blended_heat_map", show_colorbar=False, use_pyplot=False)
# # Deeplift
# dl = DeepLift(model)
# attr_dl = attribute_image_features(dl, input)
# attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

# print('Leaf is ', classes[predicted[ind]],
#       'with a Probability of:', torch.max(F.softmax(outputs, 1)).item())
