""""Classifies images from an already learned database"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from skimage.io import imshow
from skimage.io import imread

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from pickle import dump, load

from library.utils import medial_axis_skeleton
from library.utils import skeleton_lines

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

from library.data_utils import load_labels
from library.feature_extraction import extract_features
from library.feature_extraction import preprocess_image

from sys import argv

if __name__ == '__main__':
    labels = load_labels("database", "classes.csv")

    # load model
    X = np.array([])
    y = np.array(labels)
    X = load(open("X", "rb"))
    y = load(open("y", "rb"))

    clf = OneVsRestClassifier(GaussianNB())
    clf.fit(X, y)
    #proba = clf.predict_proba()

    #print(proba[0])

    # Process image
    if len(argv) < 2:
        print("Error: no pgm file given")
        exit()
    else:
        image = imread(argv[1], as_grey=True)
        p_image = preprocess_image(image)
        features = extract_features(p_image)

        proba = clf.predict_proba([features])
        for p in proba[0]:
            print("%.13f" % p)
    

    
