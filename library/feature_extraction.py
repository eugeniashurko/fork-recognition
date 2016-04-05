"""."""
import numpy as np

from skimage.measure import perimeter
from skimage.measure import regionprops
from skimage.measure import label
from skimage import transform as tf


from library.utils import fill_foreground
from library.utils import pad_image
from library.utils import smooth_border
from library.utils import medial_axis_skeleton
from library.utils import curvature_splines
from library.utils import trace_border
from library.utils import skeleton_lines
from library.utils import end_points
from library.utils import branched_points
from library.utils import get_angle

import mahotas as mh


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

    im_border = [im_dense_border[i] for i in range(len(im_dense_border))] #if i % 5 == 0]

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
    # extract properties of interest
    if len(properties) > 0:
        ratio = properties[0].area / (per * per)
        solidity = properties[0].solidity
        extent = properties[0].extent
        major_axis_scaled = properties[0].major_axis_length / per
        minor_axis_scaled = properties[0].minor_axis_length / per
    else:
        ratio = 0
        solidity = 0
        extent = 0
        major_axis_scaled = 0
        minor_axis_scaled = 0
    return [ratio, solidity, extent, major_axis_scaled, minor_axis_scaled]


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

    # find pairwise angles between skeleton lines
    # angles = []
    # for line1 in lines:
    #     for line2 in lines:
    #         if line1 != line2:
    #             angles.append(get_angle(line1, line2))
    # angles_hist = np.histogram(angles, bins=5)[0]

    return np.concatenate((
        norm_frequencies,
        [len(lines)]))


# number of branches of skeleton
def n_skeleton_branches(image):
    skeleton = mh.thin(image)
    branches = end_points(skeleton)
    return (sum(sum(branches != False)))


# number of branched points of skeleton
def n_skeleton_branched_points(image):
    skeleton = mh.thin(image)
    b_points = branched_points(skeleton)
    return (sum(sum(b_points != False)))


def centriod_displacement(image):
    binary = image
    binary[binary > 0] = 1
    lbs = label(binary)

    properties = regionprops(lbs)

    box = properties[0].bbox

    height = abs(box[2] - box[0])
    width = abs(box[3] - box[1])

    centroid = np.array([properties[0].local_centroid[0],
                         properties[0].local_centroid[1]])

    scaled_centriod = np.array([centroid[0] / height, centroid[1] / width])

    scaled_dist = np.linalg.norm(np.array([0.5, 0.5]) - scaled_centriod)
    return scaled_dist


def asymmetry_measures(image):
    binary = image
    binary[binary > 0] = 1
    lbs = label(binary)
    properties = regionprops(lbs)

    original = properties[0].image * 1.
    # flips
    left_right = np.fliplr(original)
    up_dow = np.flipud(original)
    # rotations

    angles = [45, 90, 135, 180, 225, 270, 315]
    rotations = [tf.rotate(original, angle) for angle in angles]

    # distances
    lr_dist = np.linalg.norm(original - left_right, ord=-np.inf)
    ud_dist = np.linalg.norm(original - up_dow, ord=-np.inf)

    rotation_dist = []
    for rotation in rotations:
        rotation_dist.append(np.linalg.norm(original - rotation, ord=-np.inf))

    return [lr_dist, ud_dist] + rotation_dist


def extract_features(image):
    im = preprocess_image(image)
    skeleton_dist_hist = skeleton_distances_histogram(im)
    curv_hist = border_curvature_histogram(im)
    measures = shape_measures(im)
    lines_length = skeleton_lines_length_hist(im)
    branches = n_skeleton_branches(im)
    branched_points = n_skeleton_branched_points(im)
    centroid_dist = centriod_displacement(im)
    asymmetry = asymmetry_measures(im)
    features = np.concatenate((
        skeleton_dist_hist,
        curv_hist,
        measures,
        lines_length,
        [branches],
        [branched_points],
        [centroid_dist],
        asymmetry
    ))
    return features
