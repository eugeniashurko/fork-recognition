"""Library test."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from skimage.io import imshow

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from pickle import dump, load

from library.utils import trace_border
from library.utils import border_curvature
from library.utils import fill_foreground
# from library.utils import pad_image
from library.utils import smooth_border
# from library.utils import medial_axis_skeleton

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

from library.data_utils import load_database
from library.feature_extraction import extract_features


if __name__ == '__main__':
    images, labels = load_database("database", "classes.csv")

    # key = images[4]
    # key = smooth_border(key)
    # # key = fill_foreground(key)
    # key_border = trace_border(key)
    # (known_curv, curv) = border_curvature(key_border, 5)

    # fig, ax = plt.subplots()
    # ax.imshow(key, cmap=plt.cm.gray)

    # curv = np.array(curv)
    # norm_curv = (curv - curv.min()) / (curv.max() - curv.min())
    # print(curv.max())

    # colors = []
    # for v in curv:
    #     if v > 0:
    #         colors.append(1)
    #     elif v == 0:
    #         colors.append(0)
    #     else:
    #         colors.append(-1)

    # # ax.plot(
    # #     [b[1] for b in key_border], [b[0] for b in key_border],
    # #     color="b", linewidth=5)
    # ax.scatter(
    #     [b[1] for b in known_curv], [b[0] for b in known_curv], c=colors, cmap=cm.jet)
    # plt.show()

    # Tests for features extraction pipeline
    X = []
    print("FEATURE EXTRACTION....")
    for i, image in enumerate(images):
        X.append(extract_features(image))

    # -------------
    # Machine learning
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

    print("\nLinear SVC:")
    print("+++++++++++++")
    clf = OneVsRestClassifier(LinearSVC())
    print("TRAINING....")
    clf.fit(X_train, y_train)
    print("SCORE:")
    print(clf.score(X_test, y_test))
