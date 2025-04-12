import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


train_csv_path = "/content/drive/MyDrive/num_train.csv"
test_csv_path  = "/content/drive/MyDrive/num_test.csv"

def load_and_transform(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df = df.iloc[1:, 1:]
    data = df.values.astype(float)
    num_samples = data.shape[0]
    output = []
    for i in range(num_samples):
        matrix_14x256 = data[i].reshape(14, 256)
        fft_matrix = np.fft.rfft(matrix_14x256, axis=1)
        power_spectrum = np.abs(fft_matrix) ** 2
        power_spectrum = power_spectrum[:, :128]
        output.append(power_spectrum)
    return np.array(output)

y_train_csv = np.array([1]*120 + [0]*120)
y_test_csv  = np.array([1]*30  + [0]*30)

csv_channel_names = [
    "AF3",  # row 1
    "F7",   # row 2
    "FC5",  # row 3
    "AF4",  # row 4
    "F3",   # row 5
    "T7",   # row 6
    "O1",   # row 7
    "P7",   # row 8
    "O2",   # row 9
    "P8",   # row 10
    "T8",   # row 11
    "F8",   # row 12
    "FC6",  # row 13
    "F4"    # row 14
]

treatment_communities_csv = [
    ["AF3"],
    ["F7", "FC5"],
    ["AF4", "F3"],
    ["T7"],
    ["O1", "P7"],
    ["O2", "P8"],
    ["T8"],
    ["F8", "FC6"],
    ["F4"]
]

def compute_accuracy_for_partition_csv(communities, X_train, X_test, y_train, y_test):
    X_train_list = []
    X_test_list = []
    for comm in communities:
        idx = [csv_channel_names.index(ch) for ch in comm]
        comm_train = X_train[:, idx, :].mean(axis=1)
        comm_test  = X_test[:, idx, :].mean(axis=1)
        X_train_list.append(comm_train)
        X_test_list.append(comm_test)
    X_train_feats = np.concatenate(X_train_list, axis=1)
    X_test_feats  = np.concatenate(X_test_list, axis=1)
    scaler = StandardScaler()
    X_train_feats = scaler.fit_transform(X_train_feats)
    X_test_feats  = scaler.transform(X_test_feats)
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_feats)
    X_test_pca  = pca.transform(X_test_feats)
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_pca, y_train)
    y_pred = lr.predict(X_test_pca)
    return accuracy_score(y_test, y_pred)

def random_partition_14_into_9_nonempty(channels):
    while True:
        groups = [[] for _ in range(9)]
        for ch in channels:
            groups[random.randint(0, 8)].append(ch)
        if all(len(g) > 0 for g in groups):
            return groups


train_data_path = "/content/drive/MyDrive/xtrain.npy"
test_data_path  = "/content/X_test.npy"
y_train_path    = "/content/y_train (1).npy"
y_test_path     = "/content/y_test (1).npy"

npy_channel_names = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7",
    "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]

community1 = [
    'AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'CZ', 'F1', 'F2', 'F3', 'F4',
    'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
    'FCZ', 'FP1', 'FP2', 'FPZ', 'FZ'
]
community2 = [
    'C5', 'C6', 'CB1', 'CB2', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
    'CPZ', 'FT7', 'FT8', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4',
    'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
    'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8'
]

def compute_fft_amplitude(x):
    fft_complex = np.fft.rfft(x, axis=-1)
    return np.abs(fft_complex)

def compute_averaged_features_npy(X_data_fft, channels):
    idxs = [npy_channel_names.index(ch) for ch in channels]
    X_subset = X_data_fft[:, idxs, :]
    return X_subset.mean(axis=1)

def compute_accuracy_for_partition_npy(comm1, comm2):
    train_c1_feats = compute_averaged_features_npy(X_train, comm1)
    test_c1_feats = compute_averaged_features_npy(X_test, comm1)
    train_c2_feats = compute_averaged_features_npy(X_train, comm2)
    test_c2_feats = compute_averaged_features_npy(X_test, comm2)
    X_train_concat = np.concatenate([train_c1_feats, train_c2_feats], axis=1)
    X_test_concat = np.concatenate([test_c1_feats, test_c2_feats], axis=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_concat)
    X_test_scaled = scaler.transform(X_test_concat)
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

def random_partition_62_into_2(channels):
    all_chs = channels[:]
    random.shuffle(all_chs)
    k = random.randint(1, len(all_chs) - 1)
    return all_chs[:k], all_chs[k:]
