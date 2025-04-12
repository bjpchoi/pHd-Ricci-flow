import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# Data Processing Functions (62-channel EEG data)
# =============================================================================

# File Paths
train_data_path = "/content/drive/MyDrive/xtrain.npy"  # shape: (N_train, 62, T)
test_data_path  = "/content/X_test.npy"                # shape: (N_test,  62, T)
y_train_path    = "/content/y_train (1).npy"           # shape: (N_train,)
y_test_path     = "/content/y_test (1).npy"            # shape: (N_test,)

# Channel names (62 channels)
npy_channel_names = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7",
    "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]

cluster_assignment_npy = [
    2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2,
    2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

cluster0_npy = []
cluster1_npy = []
cluster2_npy = []
for i, ch_name in enumerate(npy_channel_names):
    if cluster_assignment_npy[i] == 0:
        cluster0_npy.append(ch_name)
    elif cluster_assignment_npy[i] == 1:
        cluster1_npy.append(ch_name)
    else:
        cluster2_npy.append(ch_name)

def load_npy_data(train_data_path, test_data_path, y_train_path, y_test_path):
    print("Loading NPY data...")
    X_train = np.load(train_data_path)
    X_test = np.load(test_data_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    print("NPY data loaded.")
    return X_train, X_test, y_train, y_test

def compute_fft_amplitude(x):
    fft_complex = np.fft.rfft(x, axis=-1)
    return np.abs(fft_complex)

def compute_averaged_features_npy(X_data_fft, channels, channel_names):
    idxs = [channel_names.index(ch) for ch in channels]
    X_subset = X_data_fft[:, idxs, :]
    return X_subset.mean(axis=1)

def compute_accuracy_for_3partition_npy(comm0, comm1, comm2, X_train, X_test, y_train, y_test, channel_names):
    train_c0_feats = compute_averaged_features_npy(X_train, comm0, channel_names)
    test_c0_feats = compute_averaged_features_npy(X_test, comm0, channel_names)
    train_c1_feats = compute_averaged_features_npy(X_train, comm1, channel_names)
    test_c1_feats = compute_averaged_features_npy(X_test, comm1, channel_names)
    train_c2_feats = compute_averaged_features_npy(X_train, comm2, channel_names)
    test_c2_feats = compute_averaged_features_npy(X_test, comm2, channel_names)
    X_train_concat = np.concatenate([train_c0_feats, train_c1_feats, train_c2_feats], axis=1)
    X_test_concat = np.concatenate([test_c0_feats, test_c1_feats, test_c2_feats], axis=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_concat)
    X_test_scaled = scaler.transform(X_test_concat)
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

def random_3way_partition_npy(ch_names):
    while True:
        cluster_idx = [random.randint(0, 2) for _ in ch_names]
        c0 = [ch for i, ch in enumerate(ch_names) if cluster_idx[i] == 0]
        c1 = [ch for i, ch in enumerate(ch_names) if cluster_idx[i] == 1]
        c2 = [ch for i, ch in enumerate(ch_names) if cluster_idx[i] == 2]
        if c0 and c1 and c2:
            return c0, c1, c2

# =============================================================================
# CSV Data Processing Functions (14-channel Spectral MBD data)
# =============================================================================

# File Paths
train_csv_path = "/content/drive/MyDrive/num_train.csv"
test_csv_path = "/content/drive/MyDrive/num_test.csv"

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

# Label Vectors for CSV data
y_train = np.array([1]*120 + [0]*120)
y_test = np.array([1]*30 + [0]*30)

# Channel names (14 channels)
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
assert len(csv_channel_names) == 14, "We have exactly 14 channels."

# Original Cluster Initialization Data (for CSV data)
cluster_assignment_csv = [2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]

cluster0_csv = []
cluster1_csv = []
cluster2_csv = []
for i, ch_name in enumerate(csv_channel_names):
    if cluster_assignment_csv[i] == 0:
        cluster0_csv.append(ch_name)
    elif cluster_assignment_csv[i] == 1:
        cluster1_csv.append(ch_name)
    else:
        cluster2_csv.append(ch_name)
treatment_communities_csv = [cluster0_csv, cluster1_csv, cluster2_csv]

def compute_accuracy_for_3partition_csv(communities, X_train, X_test, y_train, y_test, channel_names):
    X_train_list = []
    X_test_list = []
    for comm in communities:
        idx = [channel_names.index(ch) for ch in comm]
        cluster_train = X_train[:, idx, :].mean(axis=1)
        cluster_test = X_test[:, idx, :].mean(axis=1)
        X_train_list.append(cluster_train)
        X_test_list.append(cluster_test)
    X_train_feats = np.concatenate(X_train_list, axis=1)
    X_test_feats = np.concatenate(X_test_list, axis=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feats)
    X_test_scaled = scaler.transform(X_test_feats)
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_pca, y_train)
    y_pred = lr.predict(X_test_pca)
    return accuracy_score(y_test, y_pred)

def random_partition_14_into_3_nonempty(channels):
    while True:
        cluster_idx = [random.randint(0, 2) for _ in channels]
        g0 = [ch for i, ch in enumerate(channels) if cluster_idx[i] == 0]
        g1 = [ch for i, ch in enumerate(channels) if cluster_idx[i] == 1]
        g2 = [ch for i, ch in enumerate(channels) if cluster_idx[i] == 2]
        if g0 and g1 and g2:
            return [g0, g1, g2]
