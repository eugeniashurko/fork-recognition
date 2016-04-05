"""Library test."""
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import cm

import numpy as np
# from skimage.io import imshow

from pickle import dump, load
# from rosetta.parallel.parallel_easy import imap_easy

# from library.utils import medial_axis_skeleton
# from library.utils import skeleton_lines

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier

from library.data_utils import load_database
from library.feature_extraction import extract_features
from library.feature_extraction import preprocess_image


def custom_test(clf, X, y):
    score = 0.0
    proba = clf.predict_proba(X)
    for i, sample_proba in enumerate(proba):
        classes = np.argsort(sample_proba)[::-1][:10]
        labels = clf.classes_[classes]
        # print(y[i])
        # print(labels)
        if y[i] in labels:
            score += 1.0
    print(score)
    return score / proba.shape[0]


if __name__ == '__main__':
    images, labels = load_database("database", "classes.csv")

    # X = list(imap_easy(extract_features, images))

    # Tests for features extraction pipeline
    X = []
    # print("FEATURE EXTRACTION....")
    # for i, image in enumerate(images):
    #     # Plotting the skeletons
    #     # ----------------------
    #     # im = preprocess_image(image)
    #     # distances_on_skeleton = medial_axis_skeleton(im)
    #     # fig, (ax1, ax2) = plt.subplots(
    #     #     1, 2, figsize=(12, 8), sharex=True, sharey=True,
    #     #     subplot_kw={'adjustable': 'box-forced'})
    #     # ax1.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    #     # ax1.axis('off')
    #     # ax2.imshow(
    #     #     distances_on_skeleton,
    #     #     cmap=plt.cm.spectral,
    #     #     interpolation='nearest')
    #     # ax2.contour(im, [0.5], colors='w')
    #     # ax2.axis('off')
    #     # fig.tight_layout()
    #     # plt.savefig("skeletons/%s_%i" % (labels[i], i))
    #     # plt.close()

    #     # # Plotting the skeleton lines
    #     # im = preprocess_image(image)
    #     # distances_on_skeleton = medial_axis_skeleton(im)
    #     # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    #     # ax1.imshow(distances_on_skeleton, cmap=plt.cm.gray)
    #     # ax1.set_title('Input image')
    #     # ax1.set_axis_off()

    #     # lines = skeleton_lines(distances_on_skeleton)

    #     # for line in lines:
    #     #     p0, p1 = line
    #     #     ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))

    #     # ax2.set_title('Probabilistic Hough')
    #     # ax2.set_axis_off()
    #     # plt.savefig("lines/%s_%i" % (labels[i], i))
    #     # plt.close()

    #     X.append(extract_features(image))

    # # -------------
    # # Machine learning
    X = np.array(X)
    y = np.array(labels)

    # dump(X, open("X", "wb"))
    # dump(y, open("y", "wb"))

    X = load(open("X", "rb"))
    y = load(open("y", "rb"))

    # Scale features
    X = scale(X)

    print("DATA: %d x %d" % X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    print("\nKNeighborsClassifier (5 neighbors):")
    print("+++++++++++++++++++++++++++++++++++")
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    print("TRAINING....")
    # clf.fit(X_train, y_train)
    print("SCORE:")
    score = cross_validation.cross_val_score(
        clf, X, y, cv=5)
    print(sum(score) / float(len(score)))
    clf.fit(X_train, y_train)

    print("TEACHERS TEST:")
    print(custom_test(clf, X_test, y_test))

    print("\nKNeighborsClassifier (20 neighbors):")
    print("+++++++++++++++++++++++++++++++++++")
    clf = KNeighborsClassifier(n_neighbors=20, n_jobs=4)
    print("TRAINING....")
    # clf.fit(X_train, y_train)
    print("SCORE:")
    score = cross_validation.cross_val_score(
        clf, X, y, cv=5)
    print(sum(score) / float(len(score)))
    clf.fit(X_train, y_train)

    print("TEACHERS TEST:")
    print(custom_test(clf, X_test, y_test))

    print("\nNaive Bayes:")
    print("+++++++++++++")
    clf = OneVsRestClassifier(GaussianNB(), n_jobs=4)
    print("TRAINING....")

    print("SCORE:")
    score = cross_validation.cross_val_score(
        clf, X, y, cv=5)
    print(sum(score) / float(len(score)))
    clf.fit(X_train, y_train)

    print("TEACHERS TEST:")
    print(custom_test(clf, X_test, y_test))

    print("\nLinear SVC:")
    print("++++++++++++")

    C = 1
    clf = OneVsRestClassifier(
        SVC(kernel='linear', C=C, probability=True), n_jobs=4)

    print("SCORE:")
    score = cross_validation.cross_val_score(
        clf, X, y, cv=5)
    print(sum(score) / float(len(score)))
    clf.fit(X_train, y_train)

    print("TEACHERS TEST:")
    print(custom_test(clf, X_test, y_test))

    print("\n\nSVC RBF kernel:")
    print("++++++++++++")

    gamma = 0.1
    C = 10
    clf = OneVsRestClassifier(
        SVC(kernel='rbf', C=C, gamma=gamma, probability=True), n_jobs=4)

    print("SCORE:")
    score = cross_validation.cross_val_score(
        clf, X, y, cv=5)
    print(sum(score) / float(len(score)))
    clf.fit(X_train, y_train)

    print("TEACHERS TEST:")
    print(custom_test(clf, X_test, y_test))
