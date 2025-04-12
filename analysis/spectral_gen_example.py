import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import networkx as nx
from tqdm import tqdm

def run_npy_clustering():
    train_data_path = "/content/drive/MyDrive/xtrain.npy"
    num_samples = 405
    num_channels = 62
    time_points = 26000
    k_min = 2
    k_max = num_channels
    data = np.load(train_data_path)
    if data.shape != (num_samples, num_channels, time_points):
        raise ValueError(f"Data shape mismatch: Expected {(num_samples, num_channels, time_points)}, but got {data.shape}.")
    print("Data loaded successfully.")
    print(f"Shape: {data.shape} (samples, channels, time_points)")
    all_correlations = np.zeros((num_channels, num_channels, num_samples))
    for i in tqdm(range(num_samples), desc="Samples processed"):
        corr_matrix = np.corrcoef(data[i], rowvar=True)
        all_correlations[:, :, i] = corr_matrix
    mean_corr = np.mean(all_correlations, axis=2)
    mean_corr[mean_corr < 0] = 0.0
    np.fill_diagonal(mean_corr, 1.0)
    G = nx.Graph()
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            G.add_edge(i, j, weight=mean_corr[i, j])
    def get_communities_from_labels(labels):
        comm_dict = {}
        for node_idx, cluster_id in enumerate(labels):
            comm_dict.setdefault(cluster_id, set()).add(node_idx)
        return list(comm_dict.values())
    best_k_modularity = None
    best_modularity = float("-inf")
    modularity_values = {}
    for k in range(k_min, k_max + 1):
        spectral_model = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
        labels_k = spectral_model.fit_predict(mean_corr)
        communities_k = get_communities_from_labels(labels_k)
        Q = nx.algorithms.community.modularity(G, communities_k, weight='weight')
        modularity_values[k] = Q
        if Q > best_modularity:
            best_modularity = Q
            best_k_modularity = k
    print(f"Best k by modularity: {best_k_modularity}")
    print(f"Max modularity = {best_modularity:.4f}")
    spectral_final = SpectralClustering(n_clusters=best_k_modularity, affinity='precomputed', assign_labels='kmeans', random_state=42)
    labels_final = spectral_final.fit_predict(mean_corr)
    print("Final Cluster Assignments:")
    for node_idx, cluster_id in enumerate(labels_final):
        print(f"Channel {node_idx} -> Cluster {cluster_id}")
    ks = list(modularity_values.keys())
    Qs = list(modularity_values.values())
    plt.figure(figsize=(7,5))
    plt.plot(ks, Qs, marker='o', linestyle='-')
    plt.title("Modularity vs. Number of Clusters (k)")
    plt.xlabel("k (clusters)")
    plt.ylabel("Modularity (Q)")
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(6,5))
    plt.imshow(mean_corr, cmap='bwr', interpolation='nearest')
    plt.colorbar(label="Thresholded Correlation")
    plt.title("Mean 62x62 Correlation Matrix")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def run_csv_clustering():
    N = 14
    num_samples = 240
    embedding_dim = 256
    expected_num_features = N * embedding_dim
    csv_file_path = "drive/MyDrive/num_train.csv"
    df = pd.read_csv(csv_file_path, header=None)
    df = df.iloc[1:, :]
    data = df.iloc[:, 1:].values
    if data.shape[1] != expected_num_features:
        raise ValueError(f"Data must have {expected_num_features} features per sample, but found {data.shape[1]}.")
    if data.shape[0] != num_samples:
        raise ValueError(f"Expected {num_samples} samples, found {data.shape[0]}.")
    data = data.astype(float)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data_reshaped = data.reshape(num_samples, N, embedding_dim)
    all_correlations = np.zeros((N, N, num_samples))
    for i in tqdm(range(num_samples), desc="Computing correlations"):
        corr_matrix = np.corrcoef(data_reshaped[i], rowvar=True)
        all_correlations[:, :, i] = corr_matrix
    mean_corr = np.mean(all_correlations, axis=2)
    mean_corr[mean_corr < 0] = 0.0
    np.fill_diagonal(mean_corr, 1.0)
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=mean_corr[i, j])
    def get_communities_from_labels(labels):
        communities_dict = {}
        for node_idx, cluster_id in enumerate(labels):
            communities_dict.setdefault(cluster_id, set()).add(node_idx)
        return list(communities_dict.values())
    best_k_modularity = None
    best_modularity = float("-inf")
    modularity_values = {}
    print("===== Searching for Best k by Modularity =====")
    for k in range(2, N + 1):
        spectral_model = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
        labels_k = spectral_model.fit_predict(mean_corr)
        comm_list = get_communities_from_labels(labels_k)
        Q = nx.algorithms.community.modularity(G, comm_list, weight='weight')
        modularity_values[k] = Q
        if Q > best_modularity:
            best_modularity = Q
            best_k_modularity = k
    print(f"Best k by modularity: {best_k_modularity}")
    print(f"Max modularity: {best_modularity:.4f}\n")
    print("===== Computing Normalized Laplacian & Plotting Eigenvalue Scree =====")
    A = mean_corr.copy()
    degrees = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt_deg = 1.0 / np.sqrt(degrees)
    inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    vals, _ = np.linalg.eig(L)
    vals_sorted = np.sort(vals)
    plt.figure(figsize=(7,5))
    plt.plot(range(1, N+1), vals_sorted, marker='o', linestyle='-')
    plt.title("Eigenvalue Scree Plot (Normalized Laplacian)")
    plt.xlabel("Eigenvalue Index (1=smallest)")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
    gaps = np.diff(vals_sorted[1:])
    largest_gap_index = np.argmax(gaps)
    best_k_eig = largest_gap_index + 1
    print(f"Largest eigenvalue gap is between eigenvalues {largest_gap_index} and {largest_gap_index+1} (0-based indexing in vals_sorted).")
    print(f"Suggesting best_k_eig (by scree) = {best_k_eig}")
    print("Look for this jump in the plot to estimate k. E.g., if there's a big jump between λ_k and λ_{k+1}, that suggests k clusters.")
    spectral_scree = SpectralClustering(n_clusters=best_k_eig, affinity='precomputed', assign_labels='kmeans', random_state=42)
    labels_scree = spectral_scree.fit_predict(mean_corr)
    print("===== Final Cluster Assignments (Best k by Eigenvalue Scree) =====")
    for node_idx, cluster_id in enumerate(labels_scree):
        print(f"Node {node_idx} (Channel {node_idx+1}) -> Cluster {cluster_id}")
    plt.figure(figsize=(7,5))
    ks = list(modularity_values.keys())
    Qs = list(modularity_values.values())
    plt.plot(ks, Qs, marker='o', linestyle='-')
    plt.title("Modularity vs. Number of Clusters (k)")
    plt.xlabel("k (clusters)")
    plt.ylabel("Modularity (Q)")
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(6,5))
    plt.imshow(mean_corr, cmap='bwr', interpolation='nearest')
    plt.colorbar(label="Thresholded Correlation")
    plt.title("Mean Correlation Matrix (14x14)\n(Negative set to 0, Diagonal=1)")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.xticks(range(N), [f"C{i+1}" for i in range(N)], rotation=45)
    plt.yticks(range(N), [f"C{i+1}" for i in range(N)])
    plt.tight_layout()
    plt.show()

def main():
    run_npy_clustering()
    run_csv_clustering()

if __name__ == "__main__":
    main()
