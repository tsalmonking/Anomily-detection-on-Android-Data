import sys
sys.path.append('utils/')

import os
import base64
import pandas as pd
from data import get_train_features_labels, get_test_features_labels, m_load, m_save
from joblib import dump, load
from constants import (
    clf_path, vect_path, outlier_dir, m_clf_path,
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

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import GridSearchCV

test_sets = [1, 2, 3, 4]

train_baseline = True
train_outlier = True
train_c_model = True
compute_results = True
plot_results = True
plot_selected_features = False
features_num = -1

X_train, y_train, vect = get_train_features_labels(training_feature_files, training_labels_csv, features_num)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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
        (OneClassSVM(nu=0.3), OneClassSVM(nu=0.3)),
        (IsolationForest(random_state=0), IsolationForest(random_state=0)),
    ]

    for clf_m, clf_g in outlier_detectors:
        clf_m.fit(X_train[y_train == 1.0])
        clf_g.fit(X_train[y_train == 0.0])

        dump(clf_m, os.path.join(outlier_dir, f"{clf_m.__class__.__name__}.mw.pkl"))
        dump(clf_g, os.path.join(outlier_dir, f"{clf_g.__class__.__name__}.gw.pkl"))
        print(f"finishing saving {clf_m.__class__.__name__}")

outlier_detector_names = ['OneClassSVM', 'IsolationForest']
outlier_detectors = {}

for name in outlier_detector_names:
    clf_m = load(os.path.join(outlier_dir, f"{name}.mw.pkl"))
    clf_g = load(os.path.join(outlier_dir, f"{name}.gw.pkl"))
    outlier_detectors[name] = (clf_m, clf_g)



##########################################
#     Training the combined detector     #
##########################################

if train_c_model:
    for name in outlier_detector_names:
        clf_m = load(os.path.join(outlier_dir, f"{name}.mw.pkl"))
        clf_g = load(os.path.join(outlier_dir, f"{name}.gw.pkl"))

        outlier_detectors[name] = (clf_m, clf_g)

    meta_features = []

    for sample in X_val:
        svm_score = [s[1] for s in model.predict_proba(sample)][0]
        score_mw = outlier_detectors['IsolationForest'][1].decision_function(sample)[0]
        score_gw = outlier_detectors['IsolationForest'][0].decision_function(sample)[0]

        meta_features.append([svm_score, score_mw, score_gw])
    

    """param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }"""

    meta_classifier = SVC(probability=True, kernel='rbf')

    grid_search = GridSearchCV(estimator=meta_classifier, 
                           param_grid=param_grid, 
                           cv=5,
                           scoring='roc_auc',
                           verbose=1,
                           n_jobs=-1)

    meta_features_train = np.array(meta_features)

    #grid_search.fit(meta_features, y_val)
    #print(f"Best parameters found: {grid_search.best_params_}", f"Best cross-validation score: {grid_search.best_score_}")
    # Chosen ones are C=0.1 gamma=1, kernel=rbf"""

    #meta_classifier = grid_search.best_estimator_
    meta_classifier.fit(meta_features, y_val)
    dump(meta_classifier, m_clf_path)

else:
    meta_classifier = load(m_clf_path)


##########################################
#           Computing the results        #
##########################################

if compute_results:

    ##########################################
    # Checking performance of classification #
    ##########################################

    for test_set in test_sets:
        results_classification_file = os.path.join(results_dir, f'results_c_{test_set}.pkl')
        results_outlier_detection_file = os.path.join(results_dir, f'results_od_{test_set}.pkl')
        results_outlier_rejection_file = os.path.join(results_dir, f'results_rej_{test_set}.pkl')

        X_test, y_test = get_test_features_labels(testing_track_3_feature_files[test_set-1], testing_labels_json, vect)
        y_pred = model.predict(X_test)
        y_score = [s[1] for s in model.predict_proba(X_test)]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_s = f1_score(y_test, y_pred)

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
    #     Checking performance of meta_m     #
    ##########################################

    for test_set in test_sets:
        results_classification = m_load(os.path.join(results_dir, f'results_c_{test_set}.pkl'))

        X_test, y_test = get_test_features_labels(testing_track_3_feature_files[test_set - 1], testing_labels_json, vect)

        meta_features_test = []

        for sample in X_test:
            svm_score = [s[1] for s in model.predict_proba(sample)][0]
            score_mw = outlier_detectors['IsolationForest'][1].decision_function(sample)[0]
            score_gw = outlier_detectors['IsolationForest'][0].decision_function(sample)[0]

            meta_features_test.append([svm_score, score_mw, score_gw])

        meta_features_test = np.array(meta_features_test)

        y_pred_meta = meta_classifier.predict(meta_features_test)
        y_score_meta = meta_classifier.predict_proba(meta_features_test)[:, 1]

        precision_meta = precision_score(y_test, y_pred_meta)
        recall_meta = recall_score(y_test, y_pred_meta)
        f1_score_meta = f1_score(y_test, y_pred_meta)

        fpr_meta, tpr_meta, _ = roc_curve(y_test, y_score_meta)
        roc_auc_meta = auc(fpr_meta, tpr_meta)

        print(f"Meta-Classifier Results for Test Set {test_set}:")
        print(f"Precision: {precision_meta:.4f}, Recall: {recall_meta:.4f}, F1 Score: {f1_score_meta:.4f}, ROC AUC: {roc_auc_meta:.4f}")
        print(f"SVM Classifier Results for Test Set {test_set}:")
        print(f"Precision: {results_classification['precision']:.4f}, Recall: {results_classification['recall']:.4f}, F1 Score: {results_classification['f1_score']:.4f}, ROC AUC: {results_classification['roc_auc']:.4f}")
        

        results_meta = {
            'y_ts': y_test,
            'y_pred': y_pred_meta,
            'y_score': y_score_meta,
            'roc_auc': roc_auc_meta,
            'precision': precision_meta,
            'recall': recall_meta,
            'f1_score': f1_score_meta
        }
        m_save(results_meta, os.path.join(results_dir, f'results_meta_{test_set}.pkl'))

##########################################
#           Plotting the results         #
##########################################

if plot_results:

    ##################################################################
    #    Plotting comparison between SVM and combinator classifier   #
    ##################################################################
    classifier_results = [[], [], [], []]
    meta_classifier_results = [[], [], [], []]

    for test_set in test_sets:
        results_classification = m_load(os.path.join(results_dir, f'results_c_{test_set}.pkl'))
        results_meta = m_load(os.path.join(results_dir, f'results_meta_{test_set}.pkl'))

        classifier_results[0].append(results_classification['precision'])
        classifier_results[1].append(results_classification['recall'])
        classifier_results[2].append(results_classification['f1_score'])
        classifier_results[3].append(results_classification['roc_auc'])

        meta_classifier_results[0].append(results_meta['precision'])
        meta_classifier_results[1].append(results_meta['recall'])
        meta_classifier_results[2].append(results_meta['f1_score'])
        meta_classifier_results[3].append(results_meta['roc_auc'])

    print(classifier_results[2], meta_classifier_results[2])

    metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
    test_set_labels = ["20Q1-20Q4", "20Q3-21Q2", "21Q1-21Q4", "21Q3-22Q2"]
    colors = ['red', 'blue', 'green', 'orange']

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(14, 8))
        plt.plot(test_set_labels, classifier_results[i], '-o', label=f'Linear SVM {metric}', color=colors[i])
        plt.plot(test_set_labels, meta_classifier_results[i], '-x', label=f'Combiner {metric}', color=colors[i])

        plt.xlabel('Test Set')
        plt.title('Metric comparison')
        plt.xticks(test_set_labels)
        plt.legend(loc='best')
        plt.grid(True)

        plt.savefig(f'{metric}.png')
        plt.show()
