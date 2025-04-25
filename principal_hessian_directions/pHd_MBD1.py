import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, probplot, pearsonr
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

num_pcs_to_plot = [0, 1, 10, 50, 100, 127]
test_pc = 0

train_transformed = np.load('train_transformed.npy')
num_samples, num_rows, num_features = train_transformed.shape
print(f"Data shape: {train_transformed.shape} (samples, rows, features)")

all_components = []
all_transformers = []

for r in range(num_rows):
    data_row = train_transformed[:, r, :]
    transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    data_gaussian = transformer.fit_transform(data_row)
    all_transformers.append(transformer)
    pca = PCA(whiten=True, random_state=42)
    data_pca = pca.fit_transform(data_gaussian)
    comps = pca.components_
    signs = np.sign(comps.sum(axis=1, keepdims=True))
    signs[signs == 0] = 1
    comps = comps * signs
    all_components.append(comps)

all_components = np.array(all_components)

avg_components = np.mean(all_components, axis=0)
U, _, Vt = np.linalg.svd(avg_components, full_matrices=False)
P_avg = U @ Vt
print("Averaged PCA loadings shape:", P_avg.shape)

np.save('averaged_pca_loadings.npy', P_avg)
print("Averaged PCA loadings saved to 'averaged_pca_loadings.npy'")

plt.figure(figsize=(10,8))
sns.heatmap(P_avg, cmap='RdBu_r', center=0)
plt.title("Averaged PCA Loadings")
plt.xlabel("Original Feature Index")
plt.ylabel("Principal Component Index")
plt.tight_layout()
plt.show()

final_data_all_rows = []

for r in range(num_rows):
    print(f"\n=== Row {r} ===")
    data_row = train_transformed[:, r, :]
    data_gaussian = all_transformers[r].transform(data_row)
    data_pca_final = data_gaussian @ P_avg.T

    corr_matrix = np.corrcoef(data_pca_final.T)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0)
    plt.title(f"Correlation Heatmap of Final PCs for Row {r}")
    plt.show()

    fig, axes = plt.subplots(len(num_pcs_to_plot), 1, figsize=(8, 4*len(num_pcs_to_plot)))
    if len(num_pcs_to_plot) == 1:
        axes = [axes]

    for ax, pc_idx in zip(axes, num_pcs_to_plot):
        pc_vals = data_pca_final[:, pc_idx]
        ax.hist(pc_vals, bins=30, edgecolor='black')
        ax.set_title(f'Row {r} - PC {pc_idx} Histogram')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    pc_vals_test = data_pca_final[:, test_pc]
    plt.figure(figsize=(6,6))
    probplot(pc_vals_test, dist="norm", plot=plt)
    plt.title(f"Row {r} - Q-Q Plot for PC {test_pc}")
    plt.grid(True)
    plt.show()

    stat, pval = normaltest(pc_vals_test)
    print(f"Row {r}, PC {test_pc} Normality Test p-value: {pval:.4e}")
    if pval < 0.05:
        print("This PC is not perfectly normal.")
    else:
        print("Cannot reject normality for this PC.")

    pc_means = np.mean(data_pca_final, axis=0)
    pc_stds = np.std(data_pca_final, axis=0)
    print(f"Row {r}: PC means (should be close to 0):", pc_means[:5], "...")
    print(f"Row {r}: PC stds (should be close to 1):", pc_stds[:5], "...")
    final_data_all_rows.append(data_pca_final)
