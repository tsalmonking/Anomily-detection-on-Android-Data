import sys
sys.path.append('utils/')

import os
import base64
import pandas as pd
from data import get_train_features_labels, get_test_features_labels, m_load, m_save
from joblib import dump, load
from constants import (
    clf_path, adv_clf_path, vect_path, outlier_dir, 
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
from pyod.models.mcd import MCD
from pyod.models.cd import CD
from pyod.models.ocsvm import OCSVM
from pyod.models.lmdd import LMDD

from pyod.models.rod import ROD
from pyod.models.hbos import HBOS
from pyod.models.dif import DIF
from pyod.models.suod import SUOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.rgraph import RGraph
from pyod.models.lunar import LUNAR
from pyod.models.iforest import IForest
from pyod.models.lof import LOF

from torch_geometric.data import Data

from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM
from secml.adv.attacks import CAttackEvasionPGDLS


test_set = int(sys.argv[1])

results_classification_file = os.path.join(results_dir, f'results_c_{test_set}.pkl')
results_outlier_detection_file = os.path.join(results_dir, f'results_od_adv_{test_set}.pkl')
results_outlier_rejection_file = os.path.join(results_dir, f'results_rej_filtered_{test_set}.pkl')

train_baseline = False
train_outlier = False
compute_results = False
plot_results = False
plot_selected_features = True
features_num = 1000

X_train, y_train, vect = get_train_features_labels(training_feature_files, training_labels_csv, features_num)

if plot_selected_features:
    s_features = vect.get_feature_names_out()
    s_feature_types = [name.split('=')[0] for name in s_features]
    s_feature_counts = Counter(s_feature_types)

    type_percentages = {feature_type: (count / sum(s_feature_counts.values())) * 100 for feature_type, count in s_feature_counts.items()}

    plt.rcParams.update({'font.size': 24})

    plt.figure(figsize=(20, 12))
    bars = plt.bar(s_feature_counts.keys(), s_feature_counts.values())

    for bar, feature_type in zip(bars, type_percentages.keys()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{type_percentages[feature_type]:.2f}%', ha='center', va='bottom')

    plt.xlabel('Feature Types')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'selected training features {features_num}.pdf'))


if train_baseline:
    model = SVC(kernel='linear', probability=True, class_weight='balanced')
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
        #MCD(),
        #CD(), # --> # --> Takes too much resources to train
        #OCSVM(),
        #LMDD(), # --> Takes too much resources to train
        
        # Proximity-Based
        #ROD(), # --> Takes too much resources to train
        HBOS(),
        
        # Outlier Ensembles
        DIF(),
        #SUOD(), # --> Takes too much resources to train
        
        # Neural Networks
        SO_GAAL(),
        #MO_GAAL(),
        
        # Graph-based
        #RGraph(), # --> Takes too much resources to train
        LUNAR()

        #IForest(),
        #LOF(),
    ]

    for clf in outlier_detectors:
        clf.fit(X_train.toarray())
        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(X_train.toarray())
        print(f"{clf.__class__.__name__}: {y_train_scores}, {y_test_scores}")

        dump(clf, os.path.join(outlier_dir, f"{clf.__class__.__name__}.pkl"))
        print(f"finishing saving {clf.__class__.__name__}")
        #exit(0)

else:
    """outlier_detector_names = ['ECOD', 'COPOD', 'PCA', 'HBOS', 
                                'DIF', 'SO_GAAL', 'LUNAR']"""
    """outlier_detector_names = ['IForest', 'LOF', 'MCD', 'OCSVM']"""

    outlier_detector_names = ['ECOD', 'COPOD', 'PCA', 'HBOS', 
                                'DIF', 'SO_GAAL', 'LUNAR',
                                'IForest', 'LOF', 'MCD', 'OCSVM']


    outlier_detectors = {name: load(os.path.join(outlier_dir, f"{name}.pkl")) for name in outlier_detector_names}


##########################################
#           Computing the results        #
##########################################

if compute_results:

    ##########################################
    #           Testing the classifier      #
    ##########################################

    if test_set == 0:
        X_test, y_test = get_test_features_labels(testing_track_1_2_feature_files, None, vect)
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

    print('misclassifications: ', 
        'malware', sum((y_pred != y_test) & (y_test == 1)), 'goodware', sum((y_pred != y_test) & (y_test == 0)),
        'correct classifications', 
        'malware', sum((y_pred == y_test) & (y_test == 1)), 'goodware', sum((y_pred == y_test) & (y_test == 0))
    )

    m_save(results_classification, results_classification_file)


    ##########################################
    #     Testing the outlier detectors      #
    ##########################################    

    # Malware samples
    X_test_mw = []

    # Malware true labels
    y_test_mw_c = []

    # Malware true labels of correctly classified samples
    y_test_t_mw = []

    # Malware outlier labels (1 if correctly classified initially and then perturbed, 0 if it's incorrectly classified and not perturbed)
    y_test_mw = []

    for x, y, y_t in zip(X_test, y_test, y_pred):
        if y == 1.0:
            X_test_mw.append(x.toarray().flatten())
            y_test_mw_c.append(y)
            if y == y_t:
                y_test_t_mw.append(y)
                y_test_mw.append(1)
            else:
                y_test_mw.append(0)


    ##########################################
    #         Adversarial manipulation       #
    ##########################################  

    if os.path.exists(adv_clf_path):
        svm_secml = m_load(adv_clf_path)
    else:
        svm_secml = CClassifierSVM(kernel='linear', class_weight='balanced')
        svm_secml.fit(X_train, y_train.astype(int))
        m_save(svm_secml, adv_clf_path) 

    adv_ds_list = list()
    solver_params = {
        'max_iter': 100,
    }

    X_test_mw_filtered = CArray([x for x, y in zip(X_test_mw, y_test_mw) if y == 1])
    
    if X_test_mw_filtered.shape[0] > 0:
        test_secml_data = CDataset(X_test_mw_filtered, y_test_t_mw)
        attack = CAttackEvasionPGDLS(
            classifier=svm_secml,
            double_init_ds=test_secml_data,
            dmax=0.3,
            y_target=0,
            solver_params=solver_params
        )
        manipulated_samples = attack.run(test_secml_data.X, test_secml_data.Y.astype(int))

        idx = 0
        for i in range(len(X_test_mw)):
            if y_test_mw[i] == 1.0:
                X_test_mw[i] = manipulated_samples[2].X.tondarray()[idx]
                idx += 1

    print("done !")

    results_outlier_detection = {}

    for name, clf in outlier_detectors.items():
        y_pred = clf.predict(np.array(X_test_mw))
        y_score = clf.decision_function(np.array(X_test_mw))

        y_ts_d = y_test_mw
        y_pred_d = y_pred
        y_score_d = y_score

        fpr, tpr, thresholds = roc_curve(y_test_mw, y_score)
        roc_auc_d = auc(fpr, tpr)

        precision_d = precision_score(y_test_mw, y_pred)
        recall_d = recall_score(y_test_mw, y_pred)
        f1_score_d = f1_score(y_test_mw, y_pred)

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

    ##########################################
    #         Classification after rej       #
    ########################################## 

    results_outlier_rejection_filtered = {}


    # Composing the dataset
    X_test_o = []
    y_test_o = []

    X_test_o.extend(X_test_mw)
    y_test_o.extend(y_test_mw_c)

    for x, y in zip(X_test, y_test):
        if y == 0:
            X_test_o.append(x.toarray().flatten())
            y_test_o.append(y)

    X_test_o = np.array(X_test_o)
    y_test_o = np.array(y_test_o)

    
    # Computing the results for the classifier without rejection
    y_pred = model.predict(X_test_o)
    y_score = [s[1] for s in model.predict_proba(X_test_o)]

    fpr, tpr, thresholds = roc_curve(y_test_o, y_score)
    roc_auc = auc(fpr, tpr)
    
    precision = precision_score(y_test_o, y_pred)
    recall = recall_score(y_test_o, y_pred)
    f1_s = f1_score(y_test_o, y_pred)


    results_outlier_rejection_filtered['SVM'] = {
        'y_ts': y_test_o,
        'y_pred': y_pred,
        'y_score': y_score,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_s
    }

    print('SVM misclassifications: ',
        'malware', sum((y_pred != y_test_o) & (y_test_o == 1)), 'goodware', sum((y_pred != y_test_o) & (y_test_o == 0)), 
        'correct classifications', 
        'malware', sum((y_pred == y_test_o) & (y_test_o == 1)), 'goodware', sum((y_pred == y_test_o) & (y_test_o == 0)), 
    )

    # Computing the results for the classifier with rejection
    for name, clf in outlier_detectors.items():
        outlier_y_pred = clf.predict(X_test_o)

        X_filtered = X_test_o[outlier_y_pred == 0]
        y_filtered = y_test_o[outlier_y_pred == 0]

        y_pred_filtered = model.predict(X_filtered)
        y_score_filtered = [s[1] for s in model.predict_proba(X_filtered)]

        print(name, 'misclassifications: ',
        'malware', sum((y_pred_filtered != y_filtered) & (y_filtered == 1)), 'goodware', sum((y_pred_filtered != y_filtered) & (y_filtered == 0)), 
        'correct classifications', 
        'malware', sum((y_pred_filtered == y_filtered) & (y_filtered == 1)), 'goodware', sum((y_pred_filtered == y_filtered) & (y_filtered == 0)), 
        )

        fpr_filtered, tpr_filtered, thresholds_filtered = roc_curve(y_filtered, y_score_filtered)
        roc_auc_filtered = auc(fpr_filtered, tpr_filtered)

        precision_filtered = precision_score(y_filtered, y_pred_filtered)
        recall_filtered = recall_score(y_filtered, y_pred_filtered)
        f1_score_filtered = f1_score(y_filtered, y_pred_filtered)

        results_outlier_rejection_filtered[name] = {
            'y_ts': y_filtered,
            'y_pred': y_pred_filtered,
            'y_score': y_score_filtered,
            'roc_auc': roc_auc_filtered,
            'precision': precision_filtered,
            'recall': recall_filtered,
            'f1_score': f1_score_filtered
        }



    m_save(results_outlier_rejection_filtered, results_outlier_rejection_file)

##########################################
#           Plotting the results         #
##########################################

if plot_results:

    plt.rcParams.update({'font.size': 18})

    results_outlier_detection = m_load(results_outlier_detection_file)

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    for name, detector in results_outlier_detection.items():
        if name == 'SO_GAAL' or name == 'OCSVM':
                continue
        fpr, tpr, _ = roc_curve(detector['y_ts'], detector['y_score'])
        roc_auc = detector['roc_auc']
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{name}").plot(ax=axs)
    
    axs.set_title(f'ROC Curves - malware')
    axs.legend(loc="lower right")
    #plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'roc_adv_{test_set}_malware.pdf'))
    plt.show()

    # Plotting classification results before and after outlier rejection
    results_classification = m_load(results_classification_file)
    results_outlier_rejection_filtered = m_load(results_outlier_rejection_file)

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    for i, results in enumerate([results_outlier_rejection_filtered]):
        for name, detector in results.items():
            if name == 'SO_GAAL' or name == 'OCSVM':
                continue
            fpr, tpr, _ = roc_curve(detector['y_ts'], detector['y_score'])
            roc_auc = detector['roc_auc']
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{name}").plot(ax=axs)
        
    axs.set_title(f'ROC Curve After Outlier Rejection - classification')
    axs.legend(loc="lower right")
    #plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'roc_classification_filtered_{test_set}.pdf'))
    plt.show()
