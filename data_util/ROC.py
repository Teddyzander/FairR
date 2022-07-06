import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from data_util.fetch_data import add_noise


def plot_ROC(classifier, dir):
    noise_level = 1
    samples = len(classifier.sens_te)
    A_is_0 = np.count_nonzero(classifier.sens_te == 0)
    A_is_1 = samples - A_is_0

    X_0 = np.zeros((A_is_0, classifier.x_te.shape[1]))
    X_1 = np.zeros((A_is_1, classifier.x_te.shape[1]))
    Y_0 = np.zeros(A_is_0)
    Y_1 = np.zeros(A_is_1)
    S_0 = np.zeros(A_is_0)
    S_1 = np.zeros(A_is_1)
    # split data sets by class
    j = 0
    k = 0
    for i in range(0, samples):
        if classifier.sens_te[i] == 0:
            X_0[j, :] = classifier.x_te[i, :]
            Y_0[j] = classifier.y_te[i]
            S_0[j] = classifier.sens_te[i]
            j += 1
        else:
            X_1[k, :] = classifier.x_te[i, :]
            Y_1[k] = classifier.y_te[i]
            S_1[k] = classifier.sens_te[i]
            k += 1

    A_0_score1 = classifier.baseline_model.decision_function(add_noise(X_0, classifier.cat, iter=1,
                                                            level=noise_level)[0])
    fpr0 = dict()
    tpr0 = dict()
    roc_auc0 = dict()
    fpr0[0], tpr0[0], _ = roc_curve(Y_0[:], A_0_score1[:])
    roc_auc0[0] = auc(fpr0[0], tpr0[0])

    fpr0["micro"], tpr0["micro"], _ = roc_curve(Y_0.ravel(), A_0_score1.ravel())
    roc_auc0["micro"] = auc(fpr0["micro"], tpr0["micro"])

    plt.figure()
    lw = 1
    plt.plot(
        fpr0[0],
        tpr0[0],
        color="red",
        lw=lw,
        label="k=0.1, A=0 (area = %0.2f)" % roc_auc0[0],
    )

    A_1_score1 = classifier.baseline_model.decision_function(add_noise(X_1, classifier.cat, iter=1,
                                                            level=noise_level)[0])
    fpr1 = dict()
    tpr1 = dict()
    roc_auc1 = dict()
    fpr1[0], tpr1[0], _ = roc_curve(Y_1[:], A_1_score1[:])
    roc_auc1[0] = auc(fpr1[0], tpr1[0])

    fpr1["micro"], tpr1["micro"], _ = roc_curve(Y_1.ravel(), A_1_score1.ravel())
    roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

    plt.plot(
        fpr1[0],
        tpr1[0],
        color="blue",
        lw=lw,
        label="k=0.1, A=1 (area = %0.2f)" % roc_auc1[0],
    )

    A_0_score0 = classifier.baseline_model.decision_function(X_0)
    fpr1 = dict()
    tpr1 = dict()
    roc_auc1 = dict()
    fpr1[0], tpr1[0], _ = roc_curve(Y_0[:], A_0_score0[:])
    roc_auc1[0] = auc(fpr1[0], tpr1[0])

    fpr1["micro"], tpr1["micro"], _ = roc_curve(Y_0.ravel(), A_0_score0.ravel())
    roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

    plt.plot(
        fpr1[0],
        tpr1[0],
        color="lightcoral",
        lw=lw,
        label="k=0, A=0 (area = %0.2f)" % roc_auc1[0],
    )

    A_1_score0 = classifier.baseline_model.decision_function(X_1)
    fpr1 = dict()
    tpr1 = dict()
    roc_auc1 = dict()
    fpr1[0], tpr1[0], _ = roc_curve(Y_1[:], A_1_score0[:])
    roc_auc1[0] = auc(fpr1[0], tpr1[0])

    fpr1["micro"], tpr1["micro"], _ = roc_curve(Y_1.ravel(), A_1_score0.ravel())
    roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

    plt.plot(
        fpr1[0],
        tpr1[0],
        color="lightblue",
        lw=lw,
        label="k=0, A=1 (area = %0.2f)" % roc_auc1[0],
    )

    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve by protected class")
    plt.legend(loc="lower right")
    plt.savefig(dir + '_ROC')
