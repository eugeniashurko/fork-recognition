"""."""
import numpy as np

from library.utils import fill_foreground
from library.utils import pad_image
from library.utils import smooth_border
from library.utils import medial_axis_skeleton


def skeleton_distances_histogram(image):
    """."""
    distances_on_skeleton = medial_axis_skeleton(image)
    non_zero_dist = distances_on_skeleton[distances_on_skeleton != 0.0]
    frequencies = np.histogram(non_zero_dist, bins=50)[0]
    # normalize
    norm_frequencies = frequencies / sum(frequencies)
    return norm_frequencies


def preprocess_image(image):
    """."""
    im = pad_image(image)
    filled_image = fill_foreground(im)
    smoothed_image = smooth_border(filled_image)
    return smoothed_image
