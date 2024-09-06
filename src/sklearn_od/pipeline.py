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

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import GridSearchCV

test_sets = [1, 2, 3, 4]

train_baseline = False
train_outlier = False
compute_results = True
plot_results = True
plot_selected_features = False
features_num = -1

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
        (OneClassSVM(gamma='auto'), OneClassSVM(gamma='auto')),
        (IsolationForest(random_state=0), IsolationForest(random_state=0)),
    ]

    for clf_m, clf_g in outlier_detectors:
        clf_m.fit(X_train[y_train == 1.0])
        clf_g.fit(X_train[y_train == 0.0])

        dump(clf_m, os.path.join(outlier_dir, f"{clf_m.__class__.__name__}.mw.pkl"))
        dump(clf_g, os.path.join(outlier_dir, f"{clf_g.__class__.__name__}.gw.pkl"))
        print(f"finishing saving {clf_m.__class__.__name__}")

else:
    outlier_detector_names = ['OneClassSVM', 'IsolationForest']
    outlier_detectors = {}

    for name in outlier_detector_names:
        clf_m = load(os.path.join(outlier_dir, f"{name}.mw.pkl"))
        clf_g = load(os.path.join(outlier_dir, f"{name}.gw.pkl"))
        outlier_detectors[name] = (clf_m, clf_g)


##########################################
#           Computing the results        #
##########################################

if compute_results:
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
        #     Testing the outlier detectors      #
        ##########################################    

        results_outlier_detection = {}

        print(outlier_detectors)
        for name, (clf_m, clf_g) in outlier_detectors.items():
            y_pred_outlier = []
            y_score_outlier = []

            # Setting threshold based on 1% rejection
            rejection_threshold = 0.1

            m_th = np.sort(clf_m.decision_function(X_train[y_train == 1.0]))[int(len(y_train == 1.0) * rejection_threshold)]
            g_th = np.sort(clf_g.decision_function(X_train[y_train == 0.0]))[int(len(y_train == 0.0) * rejection_threshold)]

            for i, sample in enumerate(X_test):
                if y_pred[i] == 1.0:
                    clf = clf_m
                    th = m_th
                else:
                    clf = clf_g
                    th = g_th
                
                scr = clf.decision_function(sample)
                y_score_outlier.append(scr[0])
                y_pred_outlier.append(1 if scr > th else -1 )
            
            y_pred_outlier = np.array(y_pred_outlier)
            y_score_outlier = np.array(y_score_outlier)

            results_outlier_detection[name] = {
                'y_ts': y_test,
                'y_pred': y_pred_outlier,
                'y_score': y_score_outlier,
            }

        m_save(results_outlier_detection, results_outlier_detection_file)


        ##########################################
        #     Checking performance after rej     #
        ##########################################

        results_classification = m_load(results_classification_file)
        results_outlier_detection = m_load(results_outlier_detection_file)

        y_test = results_classification['y_ts']
        y_pred_original = results_classification['y_pred']
        y_score_original = results_classification['y_score']

        print(f"Results before rejection for Test Set {test_set}:")
        print(f"Precision: {results_classification['precision']:.4f}, Recall: {results_classification['recall']:.4f}, F1 Score: {results_classification['f1_score']:.4f}, ROC AUC: {results_classification['roc_auc']:.4f}")

        for name, detector_results in results_outlier_detection.items():
            outlier_scores = detector_results['y_score']

            outliers_mask = outlier_scores < 0

            print('rejected', sum(outliers_mask), 'rejected malware', sum(y_test[outliers_mask]), 'rejected goodware', len(y_test[outliers_mask]) - sum(y_test[outliers_mask]))

            y_test_rejected = y_test[~outliers_mask]
            y_pred_rejected = y_pred_original[~outliers_mask]
            y_score_rejected = np.array(y_score_original)[~outliers_mask]

            precision_rejected = precision_score(y_test_rejected, y_pred_rejected)
            recall_rejected = recall_score(y_test_rejected, y_pred_rejected)
            f1_score_rejected = f1_score(y_test_rejected, y_pred_rejected)

            fpr_rejected, tpr_rejected, _ = roc_curve(y_test_rejected, y_score_rejected)
            roc_auc_rejected = auc(fpr_rejected, tpr_rejected)

            results_outlier_rejection = {
                'y_ts': y_test_rejected,
                'y_pred': y_pred_rejected,
                'y_score': y_score_rejected,
                'roc_auc': roc_auc_rejected,
                'precision': precision_rejected,
                'recall': recall_rejected,
                'f1_score': f1_score_rejected,
            }

            print(f"Results after rejection with {name} for Test Set {test_set}:")
            print(f"Precision: {precision_rejected:.4f}, Recall: {recall_rejected:.4f}, F1 Score: {f1_score_rejected:.4f}, ROC AUC: {roc_auc_rejected:.4f}")

        m_save(results_outlier_rejection, results_outlier_rejection_file)


##########################################
#           Plotting the results         #
##########################################

if plot_results:

    for test_set in test_sets:
        results_classification = m_load(os.path.join(results_dir, f'results_c_{test_set}.pkl'))
        results_outlier_detection = m_load(os.path.join(results_dir, f'results_od_{test_set}.pkl'))
        results_rejected = m_load(os.path.join(results_dir, f'results_rej_{test_set}.pkl'))
        
        for name, detector_results in results_outlier_detection.items():
            outlier_scores = detector_results['y_score']
            svm_scores = results_classification['y_score']

            plt.figure(figsize=(10, 6))
            plt.scatter(svm_scores, outlier_scores, c=results_classification['y_ts'], cmap='coolwarm', edgecolors='k')
            plt.colorbar(label='Class (0 = Goodware, 1 = Malware)')
            plt.axhline(y=0, color='green', linestyle='--', label=f'{name} Decision Boundary')
            plt.title(f'Scatter plot of SVM Score vs. {name} Outlier Score')
            plt.xlabel('SVM Score')
            plt.ylabel(f'{name} Outlier Detection Score')
            plt.grid(True)

            plt.savefig(os.path.join(figures_path, f'scatter_svm_vs_{name}_outlier_{test_set}.pdf'))
            plt.show()

    
    # Plot metric
    test_set_labels = ["20Q1-20Q4", "20Q3-21Q2", "21Q1-21Q4", "21Q3-22Q2"]

    # Data before rejection
    precision_before = []
    recall_before = []
    f1_score_before = []
    roc_auc_before = []

    # Data after rejection with OneClassSVM
    precision_after_ocsvm = []
    recall_after_ocsvm = []
    f1_score_after_ocsvm = []
    roc_auc_after_ocsvm = []

    # Data after rejection with IsolationForest
    precision_after_if = []
    recall_after_if = []
    f1_score_after_if = []
    roc_auc_after_if = []

    for test_set in test_sets:
        results_classification = m_load(os.path.join(results_dir, f'results_c_{test_set}.pkl'))
        results_rejected = m_load(os.path.join(results_dir, f'results_rej_{test_set}.pkl'))

        precision_before.append(results_classification['precision'])
        recall_before.append(results_classification['recall'])
        f1_score_before.append(results_classification['f1_score'])
        roc_auc_before.append(results_classification['roc_auc'])

        precision_after_ocsvm.append(results_rejected['precision'][0])
        recall_after_ocsvm.append(results_rejected['recall'][0])
        f1_score_after_ocsvm.append(results_rejected['f1_score'][0])
        roc_auc_after_ocsvm.append(results_rejected['roc_auc'][0])

        precision_after_if.append(results_rejected['precision'][1])
        recall_after_if.append(results_rejected['recall'][1])
        f1_score_after_if.append(results_rejected['f1_score'][1])
        roc_auc_after_if.append(results_rejected['roc_auc'][1])

    # Plotting
    metrics = ["F1 Score"]
    before_data = [f1_score_before]
    ocsvm_data = [f1_score_after_ocsvm]
    if_data = [f1_score_after_if]

    plt.figure(figsize=(14, 8))

    for i, metric in enumerate(metrics):
        plt.plot(test_set_labels, before_data[i], '-o', label=f'{metric} Before Rejection', color='blue')
        plt.plot(test_set_labels, ocsvm_data[i], '-x', label=f'{metric} After Rejection (OneClassSVM)', color='red')
        plt.plot(test_set_labels, if_data[i], '-s', label=f'{metric} After Rejection (IsolationForest)', color='green')

    plt.xlabel('Test Set')
    plt.ylabel('F1 value')
    plt.title('F1 Before and After Outlier Rejection Across Test Sets')
    plt.xticks(test_set_labels)
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('metrics_comparison_manual.png')
    plt.show()