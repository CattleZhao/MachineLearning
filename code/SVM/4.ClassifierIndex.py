# coding = utf-8

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report


if __name__ == '__main__':
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_hat = np.array([1, 0, 1, 1, 1, 1])
    print('Accuracy: \t', accuracy_score(y_true, y_hat))

    precision = precision_score(y_true, y_hat)
    print('Precision: \t', precision)

    recall = recall_score(y_true, y_hat)
    print('Recall:\t', recall)

    print('f1 score: \t', f1_score(y_true, y_hat))
    print(2 * (precision * recall) / (precision + recall))

    print('F-beta：')
    for beta in np.logspace(-3, 3, num=7, base=10):
        fbeta = fbeta_score(y_true, y_hat, beta=beta)
        print('\tbeta=%9.3f\tF-beta=%.5f' % (beta, fbeta))
        #print (1+beta**2)*precision*recall / (beta**2 * precision + recall)

    print(precision_recall_fscore_support(y_true, y_hat, beta=1))
    print(classification_report(y_true, y_hat))
