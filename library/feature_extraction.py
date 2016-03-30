"""."""
import numpy as np

from sklearn.preprocessing import normalize

from skimage.measure import perimeter
from skimage.measure import regionprops
from skimage.measure import label

from library.utils import fill_foreground
from library.utils import pad_image
from library.utils import smooth_border
from library.utils import medial_axis_skeleton
from library.utils import curvature_splines
from library.utils import trace_border


def preprocess_image(image):
    """."""
    im = pad_image(image)
    filled_image = fill_foreground(im)
    smoothed_image = smooth_border(filled_image)
    return smoothed_image


def skeleton_distances_histogram(image):
    """."""
    distances_on_skeleton = medial_axis_skeleton(image)
    non_zero_dist = distances_on_skeleton[distances_on_skeleton != 0.0]
    frequencies = np.histogram(non_zero_dist, bins=5)[0]
    # normalize
    norm_frequencies = frequencies / sum(frequencies)
    # print(norm_frequencies)
    return norm_frequencies


def border_curvature_histogram(image): 
    im_dense_border = trace_border(image)

    im_border = [im_dense_border[i] for i in range(len(im_dense_border)) if i % 5 == 0]

    x_im = np.array([x for (x, y) in im_border])
    y_im = np.array([y for (x, y) in im_border])
    curvs_im = curvature_splines(x_im, y_im)
    frequencies = np.histogram(curvs_im, bins=5)[0]
    # normalize
    norm_frequencies = frequencies / sum(frequencies)
    # print(norm_frequencies)
    return norm_frequencies


def area_perimeter_ratio(image):
    # make image binary
    new_im = image
    new_im[new_im > 0] = 1
    per = perimeter(new_im)
    lbs = label(new_im)
    properties = regionprops(lbs)
    if len(properties) > 0:
        ratio = properties[0].area / (per * per)
    else:
        ratio = 0
    return ratio


def extract_features(image):
    im = preprocess_image(image)
    skeleton_dist_hist = skeleton_distances_histogram(im)
    curv_hist = border_curvature_histogram(im)
    ap_ratio = area_perimeter_ratio(im)
    features = np.concatenate((
        skeleton_dist_hist,
        curv_hist,
        [ap_ratio]
    ))
    return features
