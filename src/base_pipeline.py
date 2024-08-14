import sys
sys.path.append('utils/')

import os
import base64
import pandas as pd
from data import get_train_features_labels, get_test_features_labels, m_save, m_load
from constants import clf_path, vect_path, outlier_dir, training_feature_files, training_labels_csv, testing_track_1_2_feature_files, testing_track_3_feature_files, testing_labels_json, results_dir
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_curve, RocCurveDisplay, auc, recall_score, f1_score
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pyod.models.ecod import ECOD


test_set = 4
results_classification_file = os.path.join(results_dir, f'results_c_{test_set}.pkl')
results_outlier_detection_file = os.path.join(results_dir, f'results_od_{test_set}.pkl')
train_baseline = False
train_outlier = False
compute_results = True
plot_results = False

features_num = 10000
X_train, y_train, vect = get_train_features_labels(training_feature_files, training_labels_csv, features_num)

if train_baseline:
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    m_save(model, clf_path)
    m_save(vect, vect_path)
else:
    model = m_load(clf_path)
    vect = m_load(vect_path)


##########################################
#     Training the outlier detectors     #
##########################################

if train_outlier:
    outlier_det = [
        ECOD() 
    ]

    clf = ECOD()
    clf.fit(X_train.toarray())
    y_train_scores = clf.decision_scores_
    y_test_scores = clf.decision_function(X_train.toarray())
    print(y_train_scores, y_test_scores)

    m_save(clf, os.path.join(outlier_dir, f"{clf.__class__.__name__}.pkl"))

else:
    clf_name = 'ECOD'
    clf = m_load(os.path.join(outlier_dir, f"{clf_name}.pkl"))


##########################################
#           Computing the results        #
##########################################

if compute_results:

    ##########################################
    #           Testing the classifiers      #
    ##########################################

    X_test, y_test = get_test_features_labels(testing_track_3_feature_files[test_set-1], testing_labels_json, vect)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_s = f1_score(y_test, y_pred)

    # Save classification results to JSON files
    results_classification = {
        'y_ts': y_test,
        'y_pred': y_pred,
        'y_score': y_score,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_s
    }

    with open(results_classification_file, 'w') as f:
        json.dump(results_classification, f)


    ##########################################
    #     Testing the outlier detectors      #
    ##########################################    

    X_test_bn = []
    X_test_mw = []
    y_test_bn = []
    y_test_mw = []

    for x, y, y_t in zip(X_test, y_test, y_pred):
        if y_t == 1.0:
            X_test_mw.append(x.toarray().flatten())
            if y != y_t:
                y_test_mw.append(1)
            else:
                y_test_mw.append(0)
        else:
            X_test_bn.append(x.toarray().flatten())
            if y != y_t:
                y_test_bn.append(1)
            else:
                y_test_bn.append(0)

    y_ts_d = []
    y_pred_d = []
    y_score_d = []
    roc_auc_d = []
    precision_d = []
    recall_d = []
    f1_score_d = []

    for x, y in zip([X_test_bn, X_test_mw], [y_test_bn, y_test_mw]):
        y_test_pred, y_test_scores = clf.predict(np.array(x), return_confidence=True)
        y_ts_d.append(y)
        y_pred_d.append(y_test_pred)
        y_score_d.append(y_test_scores)

        fpr, tpr, thresholds = roc_curve(y, y_test_scores)
        roc_auc_d.append(auc(fpr, tpr))

        precision_d.append(precision_score(y, y_test_pred))
      