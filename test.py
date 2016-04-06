"""Library test."""
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import cm

import numpy as np
# from skimage.io import imshow

from pickle import dump, load

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
from sklearn.metrics import confusion_matrix

from library.data_utils import load_database
from library.feature_extraction import extract_features
# from library.feature_extraction import preprocess_image


def custom_test(clf, X, y):
    score = 0.0
    proba = clf.predict_proba(X)
    for i, sample_proba in enumerate(proba):
        classes = np.argsort(sample_proba)[::-1][:10]
        labels = clf.classes_[classes]
        if y[i] in labels:
            score += 1.0
    print(score)
    return score / proba.shape[0]


# Util for plotting the confusion matrix
# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    images, labels = load_database("database", "classes.csv")

    # Tests for features extraction pipeline
    X = []

    # print("FEATURE EXTRACTION....")
    for i, image in enumerate(images):
        X.append(extract_features(image))

    # # -------------
    # # Machine learning pasrt
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

    # Here for the selected model we plot confusion matrix
    y_pred = clf.predict(X_test)
    target_names = np.unique(labels)
    cm = confusion_matrix(y_test, y_pred, labels=target_names)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, target_names)
    plt.show()
