import os
import gdown
import zipfile


train_files_dir = "DREBIN/files/"
train_labels_dir = "DREBIN/"
train_features_path = "DREBIN/training_set_features.zip"
train_csv_path = "DREBIN/training_set.zip"

gt_path = "test_dataset/ground_truth.json"

if not os.path.exists("DREBIN/"):
    gdown.download_folder(id="118Eb_KoW6vE38aqDY0MmVfHUtLOwO8Vk")
if not os.path.exists("test_dataset/"):
    os.mkdir("test_dataset/")
if not os.path.exists("classifiers/"):
    os.mkdir("classifiers/")
if not os.path.exists("classifiers/outlier_detectors/"):
    os.mkdir("classifiers/outlier_detectors/")

if not os.path.exists(train_features_path):
    gdown.download(id="1roRQj1fZS8RT_PisoeXACEnjhViRph3H",
                   output=train_features_path)
if not os.path.exists(train_csv_path):
    gdown.download(id="1T1Tp7Fsz4Gf0IVnX4DURG2Nu5gXbHxgk",
                   output=train_csv_path)

test_features_track1_2_gw = "test_dataset/test_set_features_track_1_2_gw.zip"
test_features_track1_2_mw = "test_dataset/test_set_features_track_1_2_mw.zip"
test_features_path_1 = "test_dataset/test_set_features_round_1.zip"
test_features_path_2 = "test_dataset/test_set_features_round_2.zip"
test_features_path_3 = "test_dataset/test_set_features_round_3.zip"
test_features_path_4 = "test_dataset/test_set_features_round_4.zip"
paths = [test_features_track1_2_gw, test_features_track1_2_mw, 
test_features_path_1, test_features_path_2, 
test_features_path_3, test_features_path_4]


track1_2_gw_id = "1EvkIuVL2zkgRTVACgTyQek7Wc_d3Lmfh"
track1_2_mw_id = "1I38c0F2wF5NvG_PmiXuZpYNZDTUrdT1E"
r1_id = '133purNiKksl_jcWZ7T2SkMaK4dJ0YNCp'
r2_id = '1BCEZcE4F0glvJlhVFnPB3Zqt6fcL_k2r'
r3_id = '114UF9erk9U0seRiqFpoc-FkxcpWRCgEC'
r4_id = '1PY3KD41JkzekXhzVau06er2VUx8iv8IN'
ids = [track1_2_gw_id, track1_2_mw_id, r1_id, r2_id, r3_id, r4_id]

for i, p in zip(ids, paths):
    if not os.path.exists(p):
        gdown.download(id=i, output=p)

with zipfile.ZipFile(train_features_path, 'r') as zip_ref:
    zip_ref.extractall(train_files_dir)
os.remove(train_features_path)

with zipfile.ZipFile(train_csv_path, 'r') as zip_ref:
    zip_ref.extractall(train_labels_dir)
os.remove(train_csv_path)
with zipfile.ZipFile(test_features_track1_2_gw, 'r') as zip_ref:
    zip_ref.extractall('test_dataset/test_set_features_track1_2_files/')
os.remove(test_features_track1_2_gw)
with zipfile.ZipFile(test_features_track1_2_mw, 'r') as zip_ref:
    zip_ref.extractall('test_dataset/test_set_features_track1_2_files/')
os.remove(test_features_track1_2_mw)


with zipfile.ZipFile(test_features_path_1, 'r') as zip_ref:
    zip_ref.extractall('test_dataset/test_set_features_track3_r1_files/')
os.remove(test_features_path_1)
with zipfile.ZipFile(test_features_path_2, 'r') as zip_ref:
    zip_ref.extractall('test_dataset/test_set_features_track3_r2_files/')
os.remove(test_features_path_2)
with zipfile.ZipFile(test_features_path_3, 'r') as zip_ref:
    zip_ref.extractall('test_dataset/test_set_features_track3_r3_files/')
os.remove(test_features_path_3)
with zipfile.ZipFile(test_features_path_4, 'r') as zip_ref:
    zip_ref.extractall('test_dataset/test_set_features_track3_r4_files/')
os.remove(test_features_path_4)