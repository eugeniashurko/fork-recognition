"""Library test."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from skimage.io import imshow

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from pickle import dump, load

from library.utils import medial_axis_skeleton

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

from library.data_utils import load_database
from library.feature_extraction import extract_features


if __name__ == '__main__':
    images, labels = load_database("database", "classes.csv")

    # Tests for features extraction pipeline
    X = []
    print("FEATURE EXTRACTION....")
    for i, image in enumerate(images):
        # distances_on_skeleton = medial_axis_skeleton(image)
        # fig, (ax1, ax2) = plt.subplots(
        #     1, 2, figsize=(12, 8), sharex=True, sharey=True,
        #     subplot_kw={'adjustable': 'box-forced'})
        # ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        # ax1.axis('off')
        # ax2.imshow(
        #     distances_on_skeleton,
        #     cmap=plt.cm.spectral,
        #     interpolation='nearest')
        # ax2.contour(image, [0.5], colors='w')
        # ax2.axis('off')
        # fig.tight_layout()
        # plt.savefig("skeletons/%s_%i" % (labels[i], i))
        # plt.close()

        X.append(extract_features(image))

    # # -------------
    # # Machine learning
    X = np.array(X)
    y = np.array(labels)

    dump(X, open("X", "wb"))
    dump(y, open("y", "wb"))

    # X = load(open("X", "rb"))
    # y = load(open("y", "rb"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    print("\nKNeighborsClassifier:")
    print("+++++++++++++++++++++")
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=20))
    print("TRAINING....")
    clf.fit(X_train, y_train)
    print("SCORE:")
    print(clf.score(X_test, y_test))

    print("\nNaive Bayes:")
    print("+++++++++++++")
    clf = OneVsRestClassifier(GaussianNB())
    print("TRAINING....")
    clf.fit(X_train, y_train)
    print("SCORE:")
    print(clf.score(X_test, y_test))

    # RETUNS PROBA TO BELONG TO CLASSES
    print(clf.predict_proba(X_test).shape)

    print("\nLinear SVC:")
    print("+++++++++++++")
    clf = OneVsRestClassifier(LinearSVC())
    print("TRAINING....")
    clf.fit(X_train, y_train)
    print("SCORE:")
    print(clf.score(X_test, y_test))

