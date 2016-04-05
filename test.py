"""Library test."""
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import cm

import numpy as np
# from skimage.io import imshow

from pickle import dump, load
from rosetta.parallel.parallel_easy import imap_easy

# from library.utils import medial_axis_skeleton
# from library.utils import skeleton_lines

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel

from library.data_utils import load_database
from library.feature_extraction import extract_features
# from library.feature_extraction import preprocess_image


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

    # Attempt to paralellize
    # X = list(imap_easy(extract_features, images, 4, 5))

    # Tests for features extraction pipeline
    X = []
    # print("FEATURE EXTRACTION....")
    for i, image in enumerate(images):
         X.append(extract_features(image))

    # # -------------
    # # Machine learning
    X = np.array(X)
    y = np.array(labels)

    dump(X, open("X", "wb"))
    dump(y, open("y", "wb"))

    # X = load(open("X", "rb"))
    # y = load(open("y", "rb"))

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

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)
    print(clf.estimators_[0].feature_importances_)

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
