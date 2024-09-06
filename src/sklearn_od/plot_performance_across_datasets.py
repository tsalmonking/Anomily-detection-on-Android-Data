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


test_set = [1, 2, 3, 4]


train_baseline = False
plot_selected_features = True
features_num = 10000

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
#           Computing the results        #
##########################################

malware_scores = []
goodware_scores = []

for t_set in test_set:
    X_test, y_test = get_test_features_labels(testing_track_3_feature_files[t_set-1], testing_labels_json, vect)
    y_pred = model.predict(X_test)
    y_score = [s[1] for s in model.predict_proba(X_test)]

    malware_scores.append(np.mean([score for score, label in zip(y_score, y_test) if label == 1]))
    goodware_scores.append(np.mean([score for score, label in zip(y_score, y_test) if label == 0]))


##########################################
#           Plotting the results         #
##########################################

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 8))

test_set_labels = ["20Q1-20Q4", "20Q3-21Q2", "21Q1-21Q4", "21Q3-22Q2"]

plt.plot(test_set_labels, malware_scores, marker='o', label='Malware Score', color='red')
plt.plot(test_set_labels, goodware_scores, marker='o', label='Goodware Score', color='green')

plt.title('SVM Scores Progression for Malware and Goodware')
plt.xlabel('Test Set')
plt.ylabel('Average Score')
plt.ylim(0, 1)
plt.xticks(test_set_labels)
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(figures_path, 'scores_progression_malware_goodware.pdf'))
plt.show()