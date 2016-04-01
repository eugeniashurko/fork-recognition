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


# Dictionary of increments of current pixel coordinates
# for finding a neighborhood with given connectivity
CONNECTIVITY_INC = {
    4: {
        0: (0, -1),
        1: (-1, 0),
        2: (0, 1),
        3: (1, 0)
    },
    8: {
        0: (0, -1),
        1: (-1, -1),
        2: (-1, 0),
        3: (-1, 1),
        4: (0, 1),
        5: (1, 1),
        6: (1, 0),
        7: (1, -1),
    }
}


def pad_image(im):
    """Pad the image by one pixel."""
    color = im[0][0]
    padded_im = np.array(
        [np.concatenate([[color], row, [color]]) for row in im], dtype=np.uint8)
    new_row = np.array(
        [color for _ in range(padded_im.shape[1])],
        dtype=np.uint8,
        ndmin=2)
    padded_im = np.concatenate([new_row, padded_im, new_row])
    return padded_im


def trace_border(im, connectivity=4):
    """Trace border with Moore-Neighbor Tracing."""
    border = list()
    # we pad image by one pixel from all sides
    # padding ensures us not to go out of domain
    padded_im = pad_image(im)

    background_pixel = padded_im[0][0]
    current_pixel = padded_im[0][0]
    start = padded_im[0][0]

    # find starting pixel
    for i in range(padded_im.shape[0]):
        for j in range(padded_im.shape[1]):
            if padded_im[i][j] != current_pixel:
                start = (i, j)
                border.append((i, j))
                break
        else:
            continue
            # executed if the loop ended normally (no break)
        break

    # iterations of the algo
    current_pixel = start
    direction = 0
    while (True):
        if connectivity == 8:
            i = (direction + 2) % 8
            while i != (direction + 7) % 8:
                next_pixel = (
                    current_pixel[0] + CONNECTIVITY_INC[connectivity][i][0],
                    current_pixel[1] + CONNECTIVITY_INC[connectivity][i][1])
                if padded_im[next_pixel] != background_pixel:
                    current_pixel = next_pixel
                    border.append(current_pixel)
                    direction = (i + 4) % 8
                    break
                i = (i + 1) % 8
        elif connectivity == 4:
            i = (direction + 1) % 4
            while i != (direction + 4) % 8:
                next_pixel = (
                    current_pixel[0] + CONNECTIVITY_INC[connectivity][i][0],
                    current_pixel[1] + CONNECTIVITY_INC[connectivity][i][1])
                if padded_im[next_pixel] != background_pixel:
                    current_pixel = next_pixel
                    border.append(current_pixel)
                    direction = (i + 2) % 4
                    break
                i = (i + 1) % 4
        else:
            raise ValueError("Invalid connectivity specified")
    # stop condition
        if current_pixel == border[0]:
            break
    # remove padding from image (it will influence border coordinates)
    border = [(pixel[0] - 1, pixel[1] - 1) for pixel in border]
    return border


def fill_border(im, border, point, color=255):
    """Fill the border."""
    image = np.array(im)
    next_points = [point]

    temporary_color = 120

    while len(next_points) > 0:
        point = next_points.pop()
        image[point] = temporary_color

        top = (point[0] - 1, point[1])
        down = (point[0] + 1, point[1])
        left = (point[0], point[1] - 1)
        right = (point[0], point[1] + 1)

        if image[top] != temporary_color and top not in border:
            next_points.append(top)
        if image[down] != temporary_color and down not in border:
            next_points.append(down)
        if image[left] != temporary_color and left not in border:
            next_points.append(left)
        if image[right] != temporary_color and right not in border:
            next_points.append(right)
    image[image == temporary_color] = color
    return image


def fill_foreground(image):
    """."""
    img_labels = label(image)
    # label_count = np.bincount(img_labels.ravel())
    # background = np.argmax(label_count)
    background = img_labels[0][0]
    filled_image = image
    filled_image[img_labels == background] = 0
    filled_image[img_labels != background] = 255
    return filled_image


def sample_border_points(border, size):
    """Sample points from the border uniformly."""
    sample_indices = np.sort(np.random.choice(range(len(border)), size=size))
    sampled_border = [p for i, p in enumerate(border) if i in sample_indices]
    return sampled_border


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


def border_curvature(border, w=5):
    i = w
    # first_ders = []
    # second_ders = []
    known = []
    curvatures = []
    while (i != len(border) - w):
        window = [border[di] for di in range(i - w, i + w + 1)]
        X = np.sort(np.array([p[1] for p in window]))
        Y = np.array([p[0] for p in window])
        Y = np.array([y for (x, y) in sorted(zip(X, Y))])

        try:
            tck = interpolate.splrep(X, Y, s=3)
            first_der = interpolate.splev(border[i][1], tck, der=1)
            second_der = interpolate.splev(border[i][1], tck, der=2)

            if not np.isnan(first_der) and not np.isnan(second_der):
                known.append(border[i])
                curvatures.append(
                    second_der / math.pow(1 + first_der * first_der, 3. / 2.))
            # print("First der: %f" % first_der)
            # print("Second der: %f" % second_der)
        except ValueError as e:
            print(e)
        i += 1
    return (known, curvatures)


def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    x1 = fx.derivative(1)(t)
    x2 = fx.derivative(2)(t)
    y1 = fy.derivative(1)(t)
    y2 = fy.derivative(2)(t)
    curvature = (x1 * y2 - y1 * x2) / np.power(x1 ** 2 + y1 ** 2, 3 / 2)
    return curvature


def skeleton_lines(skeleton):
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
