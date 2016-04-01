"""."""
import numpy as np
import re

from os import walk
from os.path import join
from skimage.io import imread


def load_database(data_path, labels_file):
    """."""
    # Read classes from file
    labels = np.genfromtxt(labels_file, delimiter=',', dtype=str)
    # because of space after comma we read redundant empty column
    labels = np.array([l[0] for l in labels])
    print("%s LABELS" % str(np.unique(labels).shape[0]))

    files = list()
    for (dirpath, dirnames, filenames) in walk(data_path):
        files = filenames

    data_images = list()
    data_labels = list()
    label_from_name = r"([a-zA-z0-9]+)[-_]\d+.pgm"

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

    print("%d DISTINCT LABELS FOUND IN DATABASE" % np.unique(data_labels).shape)

    return (data_images, data_labels)

    
def load_labels(data_path, labels_file):
    """."""
    # Read classes from file
    labels = np.genfromtxt(labels_file, delimiter=',', dtype=str)
    # because of space after comma we read redundant empty column
    labels = np.array([l[0] for l in labels])
    #print("%s LABELS" % str(np.unique(labels).shape[0]))

    files = list()
    for (dirpath, dirnames, filenames) in walk(data_path):
        files = filenames

    data_labels = list()
    label_from_name = r"([a-zA-z0-9]+)[-_]\d+.pgm"

    # here we load all the images and find their label from filename
    for f in files:
        match = re.match(label_from_name, f)
        if match:
            label = match.groups()[0]
            if label in labels:
                data_labels.append(label)
        else:
            # if name does not match our regexp or label is not in the list
            # of classes - not read from database
            continue

    #print("%d DISTINCT LABELS FOUND IN DATABASE" % np.unique(data_labels).shape)

    return data_labels
