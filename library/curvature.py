
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (20.0, 20.0)


# In[2]:

import re

from os import walk
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow, show

# Read classes from file
labels = np.genfromtxt('../classes.csv', delimiter=',', dtype=str)
# because of space after comma we read redundant empty column
labels = np.array([l[0] for l in labels])

data_path = "../database/"
files = list()
for (dirpath, dirnames, filenames) in walk(data_path):
    files = filenames  
    
data_images = list()
data_labels = list()
label_from_name = r"([a-zA-z]+)[-_]\d+.pgm"

# here we load all the images and find their label from filename
for f in files:
    match = re.match(label_from_name, f)
    if match:
        label = match.groups()[0]
        if label in labels:
            data_labels.append(label)
            data_images.append(imread(join(data_path, f), as_grey=True))      
    else: 
        # if name does not match our regexp or label is not in the list
        # of classes - not read from database
        continue 


# In[3]:

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



# In[4]:

def preprocess_image(image):
    """."""
    im = pad_image(image)
    filled_image = fill_foreground(im)
    smoothed_image = smooth_border(filled_image)
    return smoothed_image

def preprocess_image_without_smooth(image):
    """."""
    im = pad_image(image)
    filled_image = fill_foreground(im)
    return filled_image


# In[5]:

imshow(data_images[2])


# In[6]:

car = data_images[2]
car_smooth = preprocess_image(car)


# In[7]:

imshow(car_smooth)


# In[8]:

car_border = trace_border(car_smooth)

width = 12
height = 12
fig, ax = plt.subplots()
ax.imshow(car_smooth, cmap=plt.cm.gray)
ax.plot([b[1] for b in car_border], [b[0] for b in car_border], color="r", linewidth=5)


# In[9]:

from scipy.interpolate import UnivariateSpline

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

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature


# In[10]:

x_car = np.array([x for (x,y) in car_border])
y_car = np.array([y for (x,y) in car_border])


# In[11]:

car_curvature = curvature_splines(x_car, y_car)


# In[12]:

car_curvature


# In[13]:

from sklearn.preprocessing import normalize

norm_car_curv = normalize(car_curvature, norm='max')


# In[14]:

norm_car_curv


# In[15]:

pos_car_curv = []
for x in norm_car_curv[0]:
    if x>=0:
        pos_car_curv.append(x)
    elif x<-1:
        pos_car_curv.append(1)
    else:
        pos_car_curv.append(-x)


# In[16]:

pos_car_curv


# In[17]:

colors = [(curv, 0.1, 0.1) for curv in pos_car_curv]


# In[18]:

width = 12
height = 12
fig, ax = plt.subplots()
ax.imshow(car_smooth, cmap=plt.cm.gray)

ax.scatter([b[1] for b in car_border], [b[0] for b in car_border], c=colors, s=50, marker=',')


# In[ ]:




# In[166]:

elephant = preprocess_image(data_images[4])

im = pad_image(data_images[4])
filled_image = fill_foreground(im)
imshow(filled_image)


# In[210]:

elephant2 = preprocess_image_without_smooth(data_images[32])

el2_dense_border = trace_border(elephant2)

el2_border = [el2_dense_border[i] for i in range(len(el2_dense_border)) if i%5 == 0]

x_el2 = np.array([x for (x,y) in el2_border])
y_el2 = np.array([y for (x,y) in el2_border])
curvs_el2 = curvature_splines(x_el2, y_el2)

norm_el2_curv = normalize(curvs_el2, norm='max')
pos_el2_curv = []
for x in norm_car_curv[0]:
    if x>=0:
        pos_el2_curv.append(x)
    elif x<-1:
        pos_el2_curv.append(1)
    else:
        pos_el2_curv.append(-x)

colors_el2 = [(curv, 0.1, 0.1) for curv in pos_el2_curv]


width = 12
height = 12
fig, ax = plt.subplots()
ax.imshow(elephant2, cmap=plt.cm.gray)

ax.scatter([b[1] for b in el2_border], [b[0] for b in el2_border], c=colors_el2, s=100, marker=',')


# In[203]:

imshow(data_images[32])


# In[ ]:



