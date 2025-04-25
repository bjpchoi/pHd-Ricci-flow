import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import kstest, probplot, norm
from scipy.optimize import minimize
from tqdm import tqdm
import os

os.makedirs('figures', exist_ok=True)

fs = 200.0
cutoff_freq = 75.0
random_state = 42
num_pcs_to_plot = [0, 1, 10, 50, 100, 200, 404]
test_pc = 0

X_train = np.load('X_train.npy')
num_samples, num_channels, num_points = X_train.shape
print(f"X_train shape: {X_train.shape}")

f_res = fs / num_points
cutoff_bin = int(cutoff_freq / f_res)

fft_length = num_points // 2 + 1
X_fft = np.zeros((num_samples, num_channels, fft_length), dtype=np.float64)

print("Computing FFT and power spectra...")
for i in tqdm(range(num_samples), desc='FFT computation'):
    for ch in range(num_channels):
        fft_vals = np.fft.rfft(X_train[i, ch, :])
        power_spectrum = np.abs(fft_vals)**2
        X_fft[i, ch, :] = power_spectrum

print("Original FFT shape:", X_fft.shape)

X_fft = X_fft[:, :, :cutoff_bin+1]
print("Reduced FFT shape:", X_fft.shape)

plt.figure(figsize=(12, 6))
for c in range(3):
    plt.plot(X_fft[0, c, :], label=f'Channel {c}')
plt.title('Example FFT Power Spectra (First Sample, First 3 Channels, up to 75 Hz)')
plt.xlabel('Frequency Bin')
plt.ylabel('Power')
plt.legend()
plt.tight_layout()
plt.savefig('figures/sanity_check_fft_plot_75Hz.png')
plt.show()

all_transformers = []
X_gaussianized = np.zeros_like(X_fft)

print("Gaussianizing data for each channel...")
for ch in tqdm(range(num_channels), desc='Gaussianization'):
    channel_data = X_fft[:, ch, :]
    transformer = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=random_state)
    data_gauss = transformer.fit_transform(channel_data)
    X_gaussianized[:, ch, :] = data_gauss
    all_transformers.append(transformer)

n_components = 405
all_components = []

print("Performing PCA (with whitening) on each channel...")
for ch in tqdm(range(num_channels), desc='PCA per channel'):
    channel_data = X_gaussianized[:, ch, :]
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    pca.fit(channel_data)
    comps = pca.components_
    signs = np.sign(comps.sum(axis=1, keepdims=True))
    signs[signs == 0] = 1
    comps = comps * signs
    all_components.append(comps)

all_components = np.array(all_components)
print("All components shape:", all_components.shape)

avg_components = all_components.mean(axis=0)
U, _, Vt = np.linalg.svd(avg_components, full_matrices=False)
avg_components = U @ Vt
print("Averaged components shape:", avg_components.shape)

plt.figure(figsize=(10,8))
sns.heatmap(avg_components, cmap='RdBu_r', center=0)
plt.title("Averaged PCA Loadings")
plt.xlabel("Original Feature (Frequency Bin)")
plt.ylabel("Principal Component Index")
plt.tight_layout()
plt.savefig('figures/averaged_pca_loadings.png')
plt.show()

num_features = avg_components.shape[1]
X_projected_all = np.zeros((num_samples, num_channels, n_components))
X_reconstructed_all = np.zeros((num_samples, num_channels, num_features))

print("Projecting and reconstructing data using averaged PC loadings...")
for ch in tqdm(range(num_channels), desc='Projection per channel'):
    channel_data = X_gaussianized[:, ch, :]
    X_projected = channel_data @ avg_components.T
    X_reconstructed = X_projected @ avg_components
    X_projected_all[:, ch, :] = X_projected
    X_reconstructed_all[:, ch, :] = X_reconstructed

print("X_projected_all shape:", X_projected_all.shape)

num_features = X_projected_all.shape[2]
all_corr_matrices = np.zeros((num_samples, num_features, num_features))

for i in range(num_samples):
    sample_data = X_projected_all[i, :, :]
    corr_matrix = np.corrcoef(sample_data, rowvar=False)
    all_corr_matrices[i, :, :] = corr_matrix

avg_corr_matrix = all_corr_matrices.mean(axis=0)

plt.figure(figsize=(10, 8))
plt.imshow(avg_corr_matrix, aspect='auto', origin='lower', cmap='viridis')
plt.title('Averaged Post-PCA Feature Correlation Matrix (up to 75 Hz)')
plt.colorbar(label='Correlation')
plt.xlabel('Feature (PC)')
plt.ylabel('Feature (PC)')
plt.tight_layout()
plt.savefig('figures/averaged_post_pca_feature_correlation_matrix_75Hz.png')
plt.show()

np.save('X_fft_75Hz.npy', X_fft)
np.save('avg_components_75Hz.npy', avg_components)
np.save('X_projected_all_75Hz.npy', X_projected_all)
np.save('X_reconstructed_all_75Hz.npy', X_reconstructed_all)
np.save('avg_post_pca_corr_matrix_75Hz.npy', avg_corr_matrix)

print("Data saved (up to 75 Hz).")

print("Starting the optimization step...")

num_samples, num_rows, num_features = X_projected_all.shape
final_data_all_rows = []
for r in range(num_rows):
    data_row = X_projected_all[:, r, :]
    final_data_all_rows.append(data_row)

def apply_z_score_transform(x, m, s):
    if s == 0:
        return x - m
    return (x - m) / s

def ks_avg_p_value_for_feature_across_rows(params, final_data_all_rows, f_idx):
    m, s = params
    p_list = []
    for r in range(num_rows):
        x = final_data_all_rows[r][:, f_idx]
        x_trans = apply_z_score_transform(x, m, s)
        stat, p = kstest(x_trans, 'norm')
        p_list.append(p)
    return -np.mean(p_list)

def objective_single_feature(params, final_data_all_rows, f_idx):
    return ks_avg_p_value_for_feature_across_rows(params, final_data_all_rows, f_idx)

optimized_params_per_feature = np.zeros((num_features, 2))

for f in range(num_features):
    print(f"Optimizing feature {f}...")
    combined_x = np.hstack([final_data_all_rows[r][:, f] for r in range(num_rows)])
    init_m = np.mean(combined_x)
    init_s = np.std(combined_x)
    if init_s == 0:
        init_s = 1.0
    res = minimize(objective_single_feature, [init_m, init_s],
                   args=(final_data_all_rows, f),
                   method='L-BFGS-B',
                   bounds=[(None,None),(1e-9,None)],
                   options={'maxiter':50,'disp':False})
    optimized_params_per_feature[f, :] = res.x
    final_avg_p = -res.fun
    print(f"Feature {f} optimization done. Final Avg KS p-value: {final_avg_p:.4f}, params: mean={res.x[0]:.4f}, std={res.x[1]:.4f}")

final_optimized_data_all_rows = []
for r in range(num_rows):
    data_trans = np.zeros_like(final_data_all_rows[r])
    for f in range(num_features):
        m, s = optimized_params_per_feature[f]
        x = final_data_all_rows[r][:, f]
        x_trans = apply_z_score_transform(x, m, s)
        data_trans[:, f] = x_trans
    final_optimized_data_all_rows.append(data_trans)

final_optimized_data = np.stack(final_optimized_data_all_rows, axis=1)
print("Final optimized data shape:", final_optimized_data.shape)

np.save('final_optimized_data.npy', final_optimized_data)
np.save('optimized_params_per_feature.npy', optimized_params_per_feature)

for r in range(num_rows):
    data_trans = final_optimized_data_all_rows[r]
    corr_matrix = np.corrcoef(data_trans.T)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0)
    plt.title(f"Correlation Heatmap of Final PCs for Channel {r} (Z-score Optimized)")
    plt.xlabel("PC Index")
    plt.ylabel("PC Index")
    plt.tight_layout()
    plt.savefig(f'figures/corr_heatmap_channel_{r}_optimized.png')
    plt.show()
    for pc_idx in num_pcs_to_plot:
        pc_vals = data_trans[:, pc_idx]
        plt.figure(figsize=(8,4))
        plt.hist(pc_vals, bins=30, edgecolor='black')
        plt.title(f'Channel {r} - PC {pc_idx} Histogram (Z-score Optimized)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'figures/hist_channel_{r}_pc_{pc_idx}_optimized.png')
        plt.show()
        plt.figure(figsize=(6,6))
        probplot(pc_vals, dist='norm', plot=plt)
        plt.title(f"Channel {r} - Q-Q Plot for PC {pc_idx} (Z-score Optimized)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/qqplot_channel_{r}_pc_{pc_idx}_optimized.png')
        plt.show()
    pc_vals_test = data_trans[:, test_pc]
    stat, pval = kstest(pc_vals_test, 'norm')
    print(f"Channel {r}, PC {test_pc} KS Test p-value (Z-score Optimized): {pval:.4e}")
    if pval < 0.05:
        print("This PC is not perfectly normal after optimization.")
    else:
        print("Cannot reject normality for this PC after optimization.")
    pc_means = np.mean(data_trans, axis=0)
    pc_stds = np.std(data_trans, axis=0)
    print(f"Channel {r}: First few PC means after final optimization:", pc_means[:5])
    print(f"Channel {r}: First few PC stds after final optimization:", pc_stds[:5])

print("All computations and plots completed successfully.")
