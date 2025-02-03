import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, probplot, pearsonr, kstest
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
import random
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
from scipy.optimize import minimize


num_pcs_to_plot = [0, 1, 10, 50, 100, 127]
test_pc = 0
sample_features_to_plot = [0, 10, 50, 100, 127]


train_transformed = np.load('train_transformed.npy')
data = train_transformed[:, 0, :]


num_samples, num_features = data.shape
assert num_features == 128


fig, axes = plt.subplots(1, len(sample_features_to_plot), figsize=(20, 4))
for ax, f_idx in zip(axes, sample_features_to_plot):
   vals = data[:, f_idx]
   vals = vals[~np.isnan(vals)]
   ax.hist(vals, bins=30, alpha=0.7, color='blue', edgecolor='black')
   ax.set_title(f'Feature {f_idx} Before')
   ax.set_xlabel('Value')
   ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()


corr_matrix_before = np.corrcoef(data.T)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix_before, cmap='RdBu_r', center=0)
plt.title("Correlation Heatmap Before Transformation")
plt.show()


test_feature = 0
vals = data[:, test_feature]
vals = vals[~np.isnan(vals)]
stat, p = normaltest(vals)
print(f"Before Transform - Feature {test_feature}: p={p:.4e}")
if p < 0.05:
   print("This feature is not normally distributed before transformation.")
else:
   print("Cannot reject normality before transformation (unlikely given distribution).")


plt.figure(figsize=(6,6))
probplot(vals, dist="norm", plot=plt)
plt.title(f"Q-Q Plot for Feature {test_feature} Before Transformation")
plt.grid(True)
plt.show()


transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
data_gaussian = transformer.fit_transform(data)


fig, axes = plt.subplots(1, len(sample_features_to_plot), figsize=(20, 4))
for ax, f_idx in zip(axes, sample_features_to_plot):
   vals_trans = data_gaussian[:, f_idx]
   vals_trans = vals_trans[~np.isnan(vals_trans)]
   ax.hist(vals_trans, bins=30, alpha=0.7, color='green', edgecolor='black')
   ax.set_title(f'Feature {f_idx} After')
   ax.set_xlabel('Value')
   ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()


vals_trans = data_gaussian[:, test_feature]
vals_trans = vals_trans[~np.isnan(vals_trans)]
stat, p = normaltest(vals_trans)
print(f"After Transform - Feature {test_feature}: p={p:.4e}")
if p < 0.05:
   print("This feature still not normally distributed after transformation (uncommon).")
else:
   print("Cannot reject normality after transformation - looks more Gaussian now.")


plt.figure(figsize=(6,6))
probplot(vals_trans, dist='norm', plot=plt)
plt.title(f"Q-Q Plot for Feature {test_feature} After Transformation")
plt.grid(True)
plt.show()


corr_matrix_after = np.corrcoef(data_gaussian.T)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix_after, cmap='RdBu_r', center=0)
plt.title("Correlation Heatmap After Transformation")
plt.show()


print("Pairwise Pearson correlation tests after transformation (p<0.05):")
count_significant = 0
for i in range(num_features):
   for j in range(i+1, num_features):
       vals1 = data_gaussian[:, i]
       vals2 = data_gaussian[:, j]
       mask = ~np.isnan(vals1) & ~np.isnan(vals2)
       vals1 = vals1[mask]
       vals2 = vals2[mask]


       if len(vals1) > 2 and len(vals2) > 2:
           corr, p = pearsonr(vals1, vals2)
           if p < 0.05:
               count_significant += 1


print(f"Number of significantly correlated pairs (p<0.05) after transformation: {count_significant}")


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
                  options={'maxiter':50,'disp':True})
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


for r in range(num_rows):
   print(f"\n=== Row {r} After Final Optimized Z-score Transform ===")
   data_trans = final_optimized_data_all_rows[r]
   corr_matrix = np.corrcoef(data_trans.T)
   plt.figure(figsize=(10,8))
   sns.heatmap(corr_matrix, cmap='RdBu_r', center=0)
   plt.title(f"Correlation Heatmap of Final PCs for Row {r} (Z-score Optimized)")
   plt.xlabel("PC Index")
   plt.ylabel("PC Index")
   plt.show()
   for pc_idx in num_pcs_to_plot:
       pc_vals = data_trans[:, pc_idx]
       plt.figure(figsize=(8,4))
       plt.hist(pc_vals, bins=30, edgecolor='black')
       plt.title(f'Row {r} - PC {pc_idx} Histogram (Z-score Optimized)')
       plt.xlabel('Value')
       plt.ylabel('Frequency')
       plt.show()
       plt.figure(figsize=(6,6))
       probplot(pc_vals, dist='norm', plot=plt)
       plt.title(f"Row {r} - Q-Q Plot for PC {pc_idx} (Z-score Optimized)")
       plt.grid(True)
       plt.show()
   pc_vals_test = data_trans[:, test_pc]
   stat, pval = kstest(pc_vals_test, 'norm')
   print(f"Row {r}, PC {test_pc} KS Test p-value (Z-score Optimized): {pval:.4e}")
   if pval < 0.05:
       print("This PC is not perfectly normal after optimization.")
   else:
       print("Cannot reject normality for this PC after optimization.")
   pc_means = np.mean(data_trans, axis=0)
   pc_stds = np.std(data_trans, axis=0)
   print(f"Row {r}: PC means (after final optimization) ~0:", pc_means[:5], "...")
   print(f"Row {r}: PC stds (after final optimization):", pc_stds[:5], "...")
  
final_data = np.stack(final_optimized_data_all_rows, axis=1)
np.save('train_n_final.npy', final_data)
print("Final transformed data saved to 'train_n_final.npy'.")


final_data_loaded = np.load('train_n_final.npy')
num_samples, num_rows, num_features = final_data_loaded.shape
print(f"Loaded data shape: {final_data_loaded.shape} (samples, rows, features)")


normality_pvals = np.zeros((num_rows, num_features))


print("\nComputing KS test p-values for each row and feature...")
for r in tqdm(range(num_rows), desc="Rows"):
   for f in range(num_features):
       x = final_data_loaded[:, r, f]
       stat, pval = kstest(x, 'norm', args=(0, 1))
       normality_pvals[r, f] = pval


print("KS test p-value matrix computation completed.")


print("\nPlotting KS p-value matrix...")


norm_ks = TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=1)
cmap_ks = plt.cm.RdYlGn


plt.figure(figsize=(24, 4))
img = plt.imshow(normality_pvals, aspect='auto', cmap=cmap_ks, norm=norm_ks, origin='lower')
cbar = plt.colorbar(img, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('K-S Test p-value')
cbar.ax.set_yticks([0, 0.05, 0.5, 1.0])
cbar.ax.set_yticklabels(['0', '0.05', '0.5', '1'])


plt.title("K-S Normality Test p-values (0.05 = White, Split Colorbar)")
plt.xlabel("Feature Index")
plt.ylabel("Row Index")
plt.tight_layout()
plt.show()


print("\nComputing covariance matrices for each row...")


cov_matrices = []
for r in range(num_rows):
   data_r = final_data_loaded[:, r, :]
   cov_matrix = np.cov(data_r, rowvar=False)
   cov_matrices.append(cov_matrix)


print("Covariance matrices computation completed.")


print("\nPlotting covariance matrices...")


fig, axes = plt.subplots(2, 7, figsize=(18, 7))
fig.suptitle("", fontsize=16)


norm_cov = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
cmap_cov = plt.cm.seismic


for r in range(num_rows):
   row_idx = r // 7
   col_idx = r % 7
   ax = axes[row_idx, col_idx]
   im = ax.imshow(cov_matrices[r], cmap=cmap_cov, norm=norm_cov)
   ax.set_title(f"Row {r}")
   ax.axis('off')


cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('')
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.set_ticklabels(['-2', '-1', '0', '1', '2'])


plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()


print("\nSelecting random Q-Q plots and histograms...")


random.seed(0)
selected_qqlist = []
selected_histlist = []


for _ in range(7):
   r_idx = random.randint(0, num_rows - 1)
   pc_idx = random.randint(0, num_features - 1)
   selected_qqlist.append((r_idx, pc_idx))


for _ in range(7):
   r_idx = random.randint(0, num_rows - 1)
   pc_idx = random.randint(0, num_features - 1)
   selected_histlist.append((r_idx, pc_idx))


print("Selection completed.")


print("\nPlotting Q-Q plots and histograms...")


fig, axes = plt.subplots(2, 7, figsize=(21, 6))
fig.suptitle("Q-Q Plots (Top) and Histograms (Bottom)", fontsize=16)


for i, ax in enumerate(axes[0]):
   r_idx, pc_idx = selected_qqlist[i]
   data = final_data_loaded[:, r_idx, pc_idx]
   (osm, osr), (slope, intercept, r) = probplot(data, dist='norm')
   ax.scatter(osm, osr, color='blue', label='Data', alpha=0.6)
   ax.plot(osm, intercept + slope * osm, color='red', label='Fit')
   ax.set_title(f"Row {r_idx}, PC {pc_idx} Q-Q")
   ax.grid(True)
   ax.legend()


for i, ax in enumerate(axes[1]):
   r_idx, pc_idx = selected_histlist[i]
   data = final_data_loaded[:, r_idx, pc_idx]
   ax.hist(data, bins=30, color='gray', edgecolor='black', alpha=0.7)
   ax.set_title(f"Row {r_idx}, PC {pc_idx} Histogram")
   ax.set_xlabel('Value')
   ax.set_ylabel('Frequency')


plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.show()


print("\nAll plots generated successfully.")


