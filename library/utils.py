"""Set of utils for input image processing."""
import math

import numpy as np

from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.morphology import medial_axis
from skimage.measure import label

from scipy import interpolate

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


def pad_image(im, color="b"):
    """Pad the image by one pixel."""
    if color == "b":
        padded_im = np.array(
            [np.concatenate([[0], row, [0]]) for row in im], dtype=np.uint8)
        new_row = np.zeros((1, padded_im.shape[1]), dtype=np.uint8)
        padded_im = np.concatenate([new_row, padded_im, new_row])
    else:
        raise ValueError("Padding is not implemented for thes color")
    return padded_im


def trace_border(im, connectivity=4):
    """Trace border with Moore-Neighbor Tracing."""
    border = list()
    # we pad image by one pixel from all sides
    # padding ensures us not to go out of domain
    padded_im = pad_image(im)

    background_pixel = padded_im[0][0]
    current_pixel = padded_im[0][0]
    start = None

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
    label_count = np.bincount(img_labels.ravel())
    background = np.argmax(label_count)
    filled_image = image
    filled_image[img_labels != background] = 255
    return filled_image


def sample_border_points(border, size):
    """Sample points from the border uniformly."""
    sample_indices = np.sort(np.random.choice(range(len(border)), size=size))
    sampled_border = [p for i, p in enumerate(border) if i in sample_indices]
    return sampled_border


def smooth_border(im, disk_size=5):
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