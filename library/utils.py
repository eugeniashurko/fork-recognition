"""Set of utils for input image processing."""
import mahotas as mh
import math

import numpy as np

from scipy import interpolate
from scipy.interpolate import UnivariateSpline

from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.morphology import medial_axis
from skimage.measure import label
from skimage.transform import (hough_line, probabilistic_hough_line)


def pad_image(im):
    """Pad the image by one pixel."""
    color = im[0][0]

    # New image with borders
    padded_im = np.array(
        [np.concatenate([[color], row, [color]]) for row in im], dtype=np.uint8)
    new_row = np.array(
        [color for _ in range(padded_im.shape[1])],
        dtype=np.uint8,
        ndmin=2)
    padded_im = np.concatenate([new_row, padded_im, new_row])
    
    return padded_im


def fill_foreground(image):
    """Fill background in black and everything else in white"""
    # labels are different connected components
    img_labels = label(image)

    # Label of top-right pixel is considered as background
    background = img_labels[1][-2]
    filled_image = image

    # Background is black
    filled_image[img_labels == background] = 0
     # The rest is white
    filled_image[img_labels != background] = 255
    
    return filled_image


def smooth_border(im, disk_size=7):
    """Smooth the border with median filter."""
    new_im = np.array(im)
    new_im = median(new_im, disk(disk_size))
    return new_im


def medial_axis_skeleton(im):
    """Find skeleton and the distances of points on skeleton to the border."""
    skel, distance = medial_axis(im, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel

    
def skeleton_lines(skeleton):
    """Constructs the lines of the medial axis skeleton"""
    h, theta, d = hough_line(skeleton)
    lines = probabilistic_hough_line(skeleton, threshold=10, line_length=10,
                                     line_gap=10)
    return lines


def end_points(skel):
    """Find end points of skeleton.

    From the tutorial: "Construct a graph from the skeleton image of a binary form"
    http://dip4fish.blogspot.fr/2014/05/construct-graph-from-skeleton-image-of.html
    """
    endpoint1 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [2, 1, 2]])

    endpoint2 = np.array([[0, 0, 0],
                          [0, 1, 2],
                          [0, 2, 1]])

    endpoint3 = np.array([[0, 0, 2],
                          [0, 1, 1],
                          [0, 0, 2]])

    endpoint4 = np.array([[0, 2, 1],
                          [0, 1, 2],
                          [0, 0, 0]])

    endpoint5 = np.array([[2, 1, 2],
                          [0, 1, 0],
                          [0, 0, 0]])

    endpoint6 = np.array([[1, 2, 0],
                          [2, 1, 0],
                          [0, 0, 0]])

    endpoint7 = np.array([[2, 0, 0],
                          [1, 1, 0],
                          [2, 0, 0]])

    endpoint8 = np.array([[0, 0, 0],
                          [2, 1, 0],
                          [1, 2, 0]])

    ep1 = mh.morph.hitmiss(skel, endpoint1)
    ep2 = mh.morph.hitmiss(skel, endpoint2)
    ep3 = mh.morph.hitmiss(skel, endpoint3)
    ep4 = mh.morph.hitmiss(skel, endpoint4)
    ep5 = mh.morph.hitmiss(skel, endpoint5)
    ep6 = mh.morph.hitmiss(skel, endpoint6)
    ep7 = mh.morph.hitmiss(skel, endpoint7)
    ep8 = mh.morph.hitmiss(skel, endpoint8)
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
    return ep > 0


def branched_points(skel):
    """Find branching points of skeleton.

    From the tutorial: "Construct a graph from the skeleton image of a binary form"
    http://dip4fish.blogspot.fr/2014/05/construct-graph-from-skeleton-image-of.html
    """
    X = []
    # cross X
    X0 = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    X1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
    X.append(X0)
    X.append(X1)

    # T like
    T = []
    # T0 contains X0
    T0 = np.array([[2, 1, 2],
                   [1, 1, 1],
                   [2, 2, 2]])

    T1 = np.array([[1, 2, 1],
                   [2, 1, 2],
                   [1, 2, 2]])  # contains X1

    T2 = np.array([[2, 1, 2],
                   [1, 1, 2],
                   [2, 1, 2]])

    T3 = np.array([[1, 2, 2],
                   [2, 1, 2],
                   [1, 2, 1]])

    T4 = np.array([[2, 2, 2],
                   [1, 1, 1],
                   [2, 1, 2]])

    T5 = np.array([[2, 2, 1],
                   [2, 1, 2],
                   [1, 2, 1]])

    T6 = np.array([[2, 1, 2],
                   [2, 1, 1],
                   [2, 1, 2]])

    T7 = np.array([[1, 2, 1],
                   [2, 1, 2],
                   [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)

    # Y like
    Y = []
    Y0 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [2, 1, 2]])

    Y1 = np.array([[0, 1, 0],
                   [1, 1, 2],
                   [0, 2, 1]])

    Y2 = np.array([[1, 0, 2],
                   [0, 1, 1],
                   [1, 0, 2]])

    Y3 = np.array([[0, 2, 1],
                   [1, 1, 2],
                   [0, 1, 0]])

    Y4 = np.array([[2, 1, 2],
                   [0, 1, 0],
                   [1, 0, 1]])
    Y5 = np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)

    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + mh.morph.hitmiss(skel, x)
    for y in Y:
        bp = bp + mh.morph.hitmiss(skel, y)
    for t in T:
        bp = bp + mh.morph.hitmiss(skel, t)
    return bp > 0


def get_vector(line):
    return (line[1][0] - line[0][0], line[1][1] - line[0][1])


def get_angle(line1, line2):
    v1 = get_vector(line1)
    v2 = get_vector(line2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos >= 1.0:
        cos = 1
    return math.acos(cos)
