import sys
sys.path.append('utils/')

import os
import base64
import pandas as pd
from data import get_train_features_labels, get_test_features_labels, m_load, m_save
from joblib import dump, load
from constants import (
    clf_path, vect_path, outlier_dir, 
    training_feature_files, training_labels_csv, 
    testing_track_1_2_feature_files, testing_track_3_feature_files, 
    testing_labels_json, results_dir, 
    figures_path 
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, roc_curve, 
    RocCurveDisplay, auc, 
    recall_score, f1_score
)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.models.rod import ROD
from pyod.models.hbos import HBOS
from pyod.models.dif import DIF
from pyod.models.suod import SUOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.rgraph import RGraph
from pyod.models.lunar import LUNAR

test_set = 1

results_classification_file = os.path.join(results_dir, f'results_c_{test_set}.pkl')
results_outlier_detection_file = os.path.join(results_dir, f'results_od_{test_set}.pkl')
results_outlier_rejection_file = os.path.join(results_dir, f'results_rej_{test_set}.pkl')


train_baseline = False
train_outlier = False
compute_results = True
plot_results = True
plot_selected_features = True
features_num = 1000

X_train, y_train, vect = get_train_features_labels(training_feature_files, training_labels_csv, features_num)

if plot_selected_features:
    s_features = vect.get_feature_names_out()
    s_feature_types = [name.split('=')[0] for name in s_features]
    s_feature_counts = Counter(s_feature_types)

    type_percentages = {feature_type: (count / sum(s_feature_counts.values())) * 100 for feature_type, count in s_feature_counts.items()}

    plt.figure(figsize=(14, 6))
    bars = plt.bar(s_feature_counts.keys(), s_feature_counts.values())

    for bar, feature_type in zip(bars, type_percentages.keys()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{type_percentages[feature_type]:.2f}%', ha='center', va='bottom')

    plt.xlabel('Feature Types')
    plt.ylabel('Number of Features')
    plt.savefig(os.path.join(figures_path, f'selected training features {features_num}.pdf'))


if train_baseline:
    model = SVC(kernel='linear', probability=True)
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
    outlier_detectors = [
        # Probabilistic
        ECOD(),
        COPOD(),
        # Linear Model
        PCA(),
        #KPCA(), # --> Takes too much resources to train
        # Proximity-Based
        #ROD(), # --> Takes too much resources to train
        HBOS(),
        # Outlier Ensembles
        DIF(),
        #SUOD(), # --> Takes too much resources to train
        # Neural Networks
        SO_GAAL(),
        # MO_GAAL(), --> Takes too much resources to train
        # Graph-based
        #RGraph(), # --> Takes too much resources to train
        LUNAR()
    ]

    for clf in outlier_detectors:
        clf.fit(X_train.toarray())
        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(X_train.toarray())
        print(f"{clf.__class__.__name__}: {y_train_scores}, {y_test_scores}")

        dump(clf, os.path.join(outlier_dir, f"{clf.__class__.__name__}.pkl"))
        print(f"finishing saving {clf.__class__.__name__}")
        exit(0)

else:
    outlier_detector_names = ['ECOD', 'COPOD', 'PCA', 'HBOS', 
                                'DIF', 'SO_GAAL', 'LUNAR']
    outlier_detectors = {name: load(os.path.join(outlier_dir, f"{name}.pkl")) for name in outlier_detector_names}


##########################################
#           Computing the results        #
##########################################

if compute_results:

    ##########################################
    #           Testing the classifiers      #
    ##########################################

    if test_set == 0:
        X_test, y_test = get_test_features_labels(testing_track_1_2_feature_files, testing_labels_json, vect)
    else:
        X_test, y_test = get_test_features_labels(testing_track_3_feature_files[test_set-1], testing_labels_json, vect)
    y_pred = model.predict(X_test)
    y_score = [s[1] for s in model.predict_proba(X_test)]

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

    m_save(results_classification, results_classification_file)


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


    results_outlier_detection = {}

    for name, clf in outlier_detectors.items():
        y_ts_d = []
        y_pred_d = []
        y_score_d = []
        roc_auc_d = []
        precision_d = []
        recall_d = []
        f1_score_d = []

        for x, y in zip([X_test_bn, X_test_mw], [y_test_bn, y_test_mw]):
            y_pred = clf.predict(np.array(x))
            y_score = clf.decision_function(np.array(x))

            y_ts_d.append(y)
            y_pred_d.append(y_pred)
            y_score_d.append(y_score)

            fpr, tpr, thresholds = roc_curve(y, y_score)
            roc_auc_d.append(auc(fpr, tpr))

            precision_d.append(precision_score(y, y_pred))
            recall_d.append(recall_score(y, y_pred))
            f1_score_d.append(f1_score(y, y_pred))

        results_outlier_detection[name] = {
            'y_ts': y_ts_d,
            'y_pred': y_pred_d,
            'y_score': y_score_d,
            'roc_auc': roc_auc_d,
            'precision': precision_d,
            'recall': recall_d,
            'f1_score': f1_score_d
        }

    m_save(results_outlier_detection, results_outlier_detection_file)


    results_outlier_rejection = {}

    for name, clf in outlier_detectors.items():
        outlier_y_pred = clf.predict(X_test.toarray())
        #print(sum(outlier_y_pred))
        #print(sum(y_test))
        #print('------')

        x = X_test[outlier_y_pred == 0]
        y = y_test[outlier_y_pred == 0]
        y_pred = model.predict(x)
        print(f"rejected in {test_set} with {name}: total: {len(y_test) - len(y)}, malware: {sum(y_test) - sum(y)}, goodware: {(len(y_test) - sum(y_test)) - (len(y) - sum(y))}")
        #print(sum(y_pred))
        #exit(0)
        y_score = [s[1] for s in model.predict_proba(x)]

        fpr, tpr, thresholds = roc_curve(y, y_score)
        roc_auc_r = auc(fpr, tpr)

        precision_r = precision_score(y, y_pred)
        recall_r = recall_score(y, y_pred)
        f1_score_r = f1_score(y, y_pred)

        results_outlier_rejection[name] = {
            'y_ts': y,
            'y_pred': y_pred,
            'y_score': y_score,
            'roc_auc': roc_auc_r,
            'precision': precision_r,
            'recall': recall_r,
            'f1_score': f1_score_r
        }

    m_save(results_outlier_rejection, results_outlier_rejection_file)


##########################################
#           Plotting the results         #
##########################################

if plot_results:

    # Classification plots
    results_classification = m_load(results_classification_file)
    results_outlier_rejection = m_load(results_outlier_rejection_file)
    titles = ['without rejection', 'with rejection']

    fig, axs = plt.subplots(1, 1)
    for i, results in enumerate([results_classification, results_outlier_rejection]):
        if i == 0:
            print(sum(results['y_ts']))
            fpr, tpr, _ = roc_curve(results['y_ts'], results['y_score'])
            roc_auc = results['roc_auc']
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{titles[i]}").plot(ax=axs)
            continue
        for name, detector in results.items():
            fpr, tpr, _ = roc_curve(detector['y_ts'], detector['y_score'])
            roc_auc = detector['roc_auc']
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{titles[i]} {name}").plot(ax=axs)
        
    axs.set_title(f'ROC Curve')
    axs.legend(loc="lower right")
    axs.set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'roc_classification_{test_set}.pdf'))
    plt.show()


    # Outlier detection plots
    results_outlier_detection = m_load(results_outlier_detection_file)
    titles = ['benign', 'malware']

    for i in range(2):
        fig, axs = plt.subplots(1, 1)
        for name, detector in results_outlier_detection.items():
            fpr, tpr, _ = roc_curve(detector['y_ts'][i], detector['y_score'][i])
            roc_auc = detector['roc_auc'][i]
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{name} AUC = {roc_auc:.2f}").plot(ax=axs)
        
        axs.set_title(f'ROC Curves - {titles[i].capitalize()}')
        axs.legend(loc="lower right")
        axs.set_xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'roc_{test_set}_{titles[i]}.pdf'))
        plt.show()
