import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm

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

y_train = np.array([1]*120 + [0]*120)
y_test = np.array([1]*30 + [0]*30)

print("Loading data...")
X_train = load_and_transform(train_csv_path)
X_test = load_and_transform(test_csv_path)
print("Data loaded.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}")

channel_names = [
    "AF3",
    "F7",
    "FC5",
    "AF4",
    "F3",
    "T7",
    "O1",
    "P7",
    "O2",
    "P8",
    "T8",
    "F8",
    "FC6",
    "F4"
]
assert len(channel_names) == 14, "We have exactly 14 channels."

cluster0 = []
cluster1 = []
cluster2 = []
cluster_assignment = [2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]
for i, ch_name in enumerate(channel_names):
    if cluster_assignment[i] == 0:
        cluster0.append(ch_name)
    elif cluster_assignment[i] == 1:
        cluster1.append(ch_name)
    else:
        cluster2.append(ch_name)
treatment_communities = [cluster0, cluster1, cluster2]

def compute_accuracy_for_3partition(communities, X_train, X_test, y_train, y_test):
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
    acc = accuracy_score(y_test, y_pred)
    return acc

treatment_acc = compute_accuracy_for_3partition(treatment_communities, X_train, X_test, y_train, y_test)
print(f"TREATMENT (3-group) => Test ACC = {treatment_acc:.4f}")

def random_partition_14_into_3_nonempty(channels):
    while True:
        cluster_idx = [random.randint(0, 2) for _ in channels]
        g0 = [ch for i, ch in enumerate(channels) if cluster_idx[i] == 0]
        g1 = [ch for i, ch in enumerate(channels) if cluster_idx[i] == 1]
        g2 = [ch for i, ch in enumerate(channels) if cluster_idx[i] == 2]
        if len(g0) > 0 and len(g1) > 0 and len(g2) > 0:
            return [g0, g1, g2]

n_controls = 19
control_accs = []
print("\nComputing 19 random partitions into 3 groups (CONTROL)...")
for _ in tqdm(range(n_controls)):
    communities = random_partition_14_into_3_nonempty(channel_names)
    acc = compute_accuracy_for_3partition(communities, X_train, X_test, y_train, y_test)
    control_accs.append(acc)

count_ge = sum(1 for acc in control_accs if acc >= treatment_acc)
p_value = count_ge / n_controls

print("\n==========================================")
print("TREATMENT PARTITION (3 Groups) => channels:")
print(f"Cluster 0: {cluster0}")
print(f"Cluster 1: {cluster1}")
print(f"Cluster 2: {cluster2}")
print(f"Treatment ACC = {treatment_acc:.4f}")
print("==========================================")
print("CONTROL PARTITIONS ACCURACIES:")
for i, acc in enumerate(control_accs, start=1):
    print(f"{i:2d}) {acc:.4f}")
print("------------------------------------------")
print(f"Number of control partitions: {n_controls}")
print(f"One-sided p-value = {p_value:.4f}")
print("==========================================")
