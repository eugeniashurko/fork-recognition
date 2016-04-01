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
from library.utils import skeleton_lines


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
    frequencies = np.histogram(non_zero_dist, bins=10)[0]
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


def shape_measures(image):
    # make image binary
    new_im = image
    new_im[new_im > 0] = 1
    per = perimeter(new_im)
    lbs = label(new_im)
    properties = regionprops(lbs)

    # area / perimeter ratio
    if len(properties) > 0:
        ratio = properties[0].area / (per * per)
        solidity = properties[0].solidity
    else:
        ratio = 0
        solidity = 0

    # solidity
    return [ratio, solidity]


def skeleton_lines_length_hist(image):
    skeleton = medial_axis_skeleton(image)
    lines = skeleton_lines(skeleton)
    lenghts = np.array([np.linalg.norm(
        np.array(line[0]) - np.array(line[1])) for line in lines])
    frequencies = np.histogram(lenghts, bins=5)[0]
    # normalize
    if sum(frequencies) != 0:
        norm_frequencies = frequencies / sum(frequencies)
    else:
        norm_frequencies = np.array([0 for i in range(frequencies.shape[0])])
    # print(norm_frequencies)
    return norm_frequencies

# number of branches of skeleton
def n_skeleton_branches(image):
    skeleton = medial_axis_skeleton(image)
    peaks = 0
    skeleton[skeleton != 0] = 1
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            skeleton_pixels_in_neighborhood =\
                skeleton[i - 1][j - 1] +\
                skeleton[i - 1][j] +\
                skeleton[i - 1][j + 1] +\
                skeleton[i][j - 1] +\
                skeleton[i][j] +\
                skeleton[i][j + 1] +\
                skeleton[i][j - 1] +\
                skeleton[i][j] +\
                skeleton[i][j + 1]
            if skeleton_pixels_in_neighborhood == 1:
                peaks += 1
    return peaks


def extract_features(image):
    im = preprocess_image(image)
    skeleton_dist_hist = skeleton_distances_histogram(im)
    curv_hist = border_curvature_histogram(im)
    measures = shape_measures(im)
    lines_length = skeleton_lines_length_hist(im)
    # branches = n_skeleton_branches(im)
    features = np.concatenate((
        skeleton_dist_hist,
        curv_hist,
        # measures,
        # lines_length,
        # [branches]
    ))
    return features
