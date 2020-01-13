from PIL import Image as PImage
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift
from captum.attr import Saliency
from captum.attr import visualization as viz
from captum.attr import GuidedGradCam
from captum.attr._core.guided_grad_cam import LayerGradCam
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import feature
from spectralloader import Spectralloader



# create explainers for given image
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

    def normalize(data):
        # consider only the positive values
        for i in range(h):
            for k in range(w):
                for j in range(c):
                    if data[i][k][j] < 0:
                        data[i][k][j] = 0
        # reshape to hxw
        d_img = data[:, :, 0] + data[:, :, 1] + data[:, :, 2]
        max = np.max(d_img)
        mean = np.mean(d_img)
        min = 0
        designated_mean = 0.25
        factor = (designated_mean * max) / mean
        # print('max val: ' + str(max) + ' mean val: ' + str(mean) + ' faktor: ' + str(factor))
        # normalize
        for i in range(h):
            for k in range(w):
                d_img[i][k] = (d_img[i][k] - min) / (max - min)
                if d_img[i][k] > 1:
                    d_img[i][k] = 1
        d_img = ndi.gaussian_filter(d_img, 8)

        return d_img

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    # # saliency
    # saliency = Saliency(model)
    # grads = saliency.attribute(input, target=label)
    # grads = normalize(np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0)))
    #
    # # IntegratedGradients
    ig = IntegratedGradients(model)
    # attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    # attr_ig = normalize(np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)))
    #
    # # IntegratedGradients Noise Tunnel
    # nt = NoiseTunnel(ig)
    # attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
    #                                       n_samples=5,
    #                                       # stdevs=0.2
    #                                       )
    #
    # attr_ig_nt = normalize(np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
    #
    # # GuidedGradCam
    # gc = GuidedGradCam(model, model.layer4)
    # attr_gc = attribute_image_features(gc, input)
    # attr_gc = normalize(np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
    #
    # # GradCam Original Layer 4
    # gco = LayerGradCam(model, model.layer4)
    # attr_gco = attribute_image_features(gco, input)
    #
    # # GradCam
    # att = attr_gco.squeeze(0).squeeze(0).cpu().detach().numpy()
    # # gco_int = (att * 255).astype(np.uint8)
    # gradcam = PImage.fromarray(att).resize((w, h), PImage.ANTIALIAS)
    # np_gradcam = np.asarray(gradcam)
    #
    # f2 = detect_edge(grads)
    # f3 = detect_edge(attr_ig)
    # f4 = detect_edge(attr_ig_nt)
    # f6 = detect_edge(attr_gc)
    # f7 = f8 = detect_edge(np_gradcam)

    # IntegratedGradients Noise Tunnel
    nt = NoiseTunnel(ig)
    attr_ig_nt1 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=5,
                                          # stdevs=0.2
                                          )

    # IntegratedGradients Noise Tunnel
    attr_ig_nt2 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=3,
                                          # stdevs=0.2
                                          )
    # IntegratedGradients Noise Tunnel
    attr_ig_nt3 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=5,
                                          # stdevs=0.2
                                          )
    # IntegratedGradients Noise Tunnel
    attr_ig_nt4 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=8,
                                          # stdevs=0.2
                                          )
    # IntegratedGradients Noise Tunnel
    attr_ig_nt5 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=5,
                                          stdevs=0.5
                                          )
    # IntegratedGradients Noise Tunnel
    attr_ig_nt6 = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          n_samples=5,
                                          stdevs=1.0
                                          )
    f2 = detect_edge(attr_ig_nt1)
    f3 = detect_edge(attr_ig_nt2)
    f4 = detect_edge(attr_ig_nt3)
    f6 = detect_edge(attr_ig_nt4)
    f7 = detect_edge(attr_ig_nt5)
    f8 = detect_edge(attr_ig_nt6)




    # original_image = detect_edge()
    # # Original Image
    f1, a1 = viz.visualize_image_attr(None, np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      method="original_image", use_pyplot=False)
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
    return [f1, f2, f3, f4, f6, f7, f8]