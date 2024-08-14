import os
import json
import base64
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pickle


def get_train_features_labels(feature_files, labels, k):
    df = pd.read_csv(labels)
    df['features'] = df['sha256'].apply(lambda sha256_name: json.load(open(os.path.join(feature_files, f"{sha256_name.upper()}.json"), 'r')))

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(df['features'])
    y = df['label'].values

    selector = SelectKBest(score_func=chi2, k=k).fit(X, y)
    vectorizer = vectorizer.restrict(selector.get_support())
    X = vectorizer.transform(df['features'])

    return X, y, vectorizer


def get_test_features_labels(feature_files, labels, vect):
    with open(labels, 'r') as f:
        labels_dict = json.load(f)

    labels_dict = {k: v for d in labels_dict for k, v in d.items()}
    feature_files_list = [os.path.splitext(f)[0].lower() for f in os.listdir(feature_files) if f.endswith('.json')]
    filtered_labels_dict = {k: labels_dict[k] for k in feature_files_list if k in labels_dict}
    df = pd.DataFrame(list(filtered_labels_dict.items()), columns=['sha256', 'label'])
    df['features'] = df['sha256'].apply(lambda sha256_name: json.load(open(os.path.join(feature_files, f"{sha256_name.upper()}.json"), 'r')))

    X = vect.transform(df['features'])
    y = df['label'].values

    return X, y


def m_load(path, format='rb', module=pickle):
    with open(path, format) as f:
        object = module.load(f)
    return object


def m_save(object, path, format='wb', module=pickle):
    with open(path, format) as f:
        module.dump(object, f)