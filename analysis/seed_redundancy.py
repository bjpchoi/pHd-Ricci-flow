import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

avg_pairs = [
    ("AF3", "F1"),
    ("F5", "FC1"),
    ("C3", "F8"),
    ("C3", "FZ"),
    ("F8", "FZ"),
    ("OZ", "P2"),
    ("OZ", "P4"),
    ("OZ", "PO6"),
    ("P2", "P4"),
    ("P2", "PO6"),
    ("P4", "PO6"),
    ("CP2", "FT7"),
    ("C6", "CB2"),
    ("C6", "P3"),
    ("C6", "PZ"),
    ("CB2", "P3"),
    ("CB2", "PZ"),
    ("P3", "PZ"),
    ("P7", "PO8")
]

N_CONTROL_PAIRS = 19
train_data_path = "/content/drive/MyDrive/xtrain.npy"
test_data_path = "/content/X_test.npy"
y_train_path = "/content/y_train (1).npy"
y_test_path = "/content/y_test (1).npy"

channel_names = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7",
    "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]

N_PERMUTATIONS = 100000

print("Loading data...")
X_train = np.load(train_data_path)
X_test = np.load(test_data_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)
print("Data loaded.")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

all_possible_pairs = []
for i in range(len(channel_names)):
    for j in range(i + 1, len(channel_names)):
        all_possible_pairs.append((channel_names[i], channel_names[j]))
print(f"Total possible pairs from 62 channels: {len(all_possible_pairs)}")

avg_pairs_set = set(avg_pairs)
candidate_control_pairs = [p for p in all_possible_pairs if p not in avg_pairs_set]
print(f"After removing the fixed avg_pairs, we have {len(candidate_control_pairs)} possible pairs left.")

rng = np.random.default_rng(seed=42)
chosen_control_idx = rng.choice(len(candidate_control_pairs), size=N_CONTROL_PAIRS, replace=False)
control_pairs = [candidate_control_pairs[k] for k in chosen_control_idx]
print(f"Randomly selected {N_CONTROL_PAIRS} distinct 'control' pairs.")

all_38_pairs = avg_pairs + control_pairs
print("\nFinal Pair Groups:")
print("-" * 50)
print("Avg Pairs (fixed):")
for p in avg_pairs:
    print("  ", p)
print("-" * 50)
print("Control Pairs (randomly chosen):")
for p in control_pairs:
    print("  ", p)

def get_time_domain_features_for_pair(X, ch_names, pair, method='separate'):
    ch_idx = [ch_names.index(ch) for ch in pair]
    subset = X[:, ch_idx, :]
    if method == 'separate':
        return subset.reshape(subset.shape[0], -1)
    elif method == 'average':
        return subset.mean(axis=1)
    else:
        raise ValueError("method must be 'separate' or 'average'")

def compute_sep_avg_accuracies(pair):
    X_train_sep = get_time_domain_features_for_pair(X_train, channel_names, pair, 'separate')
    X_test_sep = get_time_domain_features_for_pair(X_test, channel_names, pair, 'separate')
    pca_sep = PCA(n_components=50)
    X_train_sep_pca = pca_sep.fit_transform(X_train_sep)
    X_test_sep_pca = pca_sep.transform(X_test_sep)
    scaler_sep = StandardScaler()
    X_train_sep_pca_scaled = scaler_sep.fit_transform(X_train_sep_pca)
    X_test_sep_pca_scaled = scaler_sep.transform(X_test_sep_pca)
    lr_sep = LogisticRegression(max_iter=100000, solver='saga')
    lr_sep.fit(X_train_sep_pca_scaled, y_train)
    sep_acc = accuracy_score(y_test, lr_sep.predict(X_test_sep_pca_scaled))

    X_train_avg = get_time_domain_features_for_pair(X_train, channel_names, pair, 'average')
    X_test_avg = get_time_domain_features_for_pair(X_test, channel_names, pair, 'average')
    pca_avg = PCA(n_components=50)
    X_train_avg_pca = pca_avg.fit_transform(X_train_avg)
    X_test_avg_pca = pca_avg.transform(X_test_avg)
    scaler_avg = StandardScaler()
    X_train_avg_pca_scaled = scaler_avg.fit_transform(X_train_avg_pca)
    X_test_avg_pca_scaled = scaler_avg.transform(X_test_avg_pca)
    lr_avg = LogisticRegression(max_iter=100000, solver='saga')
    lr_avg.fit(X_train_avg_pca_scaled, y_train)
    avg_acc = accuracy_score(y_test, lr_avg.predict(X_test_avg_pca_scaled))
    return (sep_acc, avg_acc)

def rel_diff(sep_acc, avg_acc):
    if sep_acc == 0:
        return np.nan
    return (sep_acc - avg_acc) / sep_acc

print("\nComputing logistic regression (PCA=50) accuracies for the 38 pairs...")
pair2results = {}
for pair in tqdm(all_38_pairs, desc="Regression loop"):
    pair2results[pair] = compute_sep_avg_accuracies(pair)

pair2reldiff = {p: rel_diff(*pair2results[p]) for p in all_38_pairs}
avg_reldiffs = [pair2reldiff[p] for p in avg_pairs]
control_reldiffs = [pair2reldiff[p] for p in control_pairs]
mean_avg_reldiff = np.nanmean(avg_reldiffs)
mean_control_reldiff = np.nanmean(control_reldiffs)
observed_diff = mean_avg_reldiff - mean_control_reldiff

print("\nObserved Results (Using PCA=50):")
print(f"  Mean RelDiff (avg_pairs)     = {mean_avg_reldiff:.4f}")
print(f"  Mean RelDiff (control_pairs) = {mean_control_reldiff:.4f}")
print(f"  Observed difference (avg - control) = {observed_diff:.4f}")

all_reldiffs = np.array([pair2reldiff[p] for p in all_38_pairs])
N = len(all_reldiffs)
n_avg = len(avg_pairs)
n_control = len(control_pairs)
perm_diffs = []
print(f"\nBuilding null distribution with {N_PERMUTATIONS} permutations...")

for _ in tqdm(range(N_PERMUTATIONS), desc="Permutation loop"):
    perm_idx = rng.permutation(N)
    avg_idx = perm_idx[:n_avg]
    cont_idx = perm_idx[n_avg:]
    mean_avg = np.nanmean(all_reldiffs[avg_idx])
    mean_cont = np.nanmean(all_reldiffs[cont_idx])
    perm_diffs.append(mean_avg - mean_cont)

perm_diffs = np.array(perm_diffs)
abs_observed = abs(observed_diff)
abs_perms = np.abs(perm_diffs)
p_value = np.mean(abs_perms >= abs_observed)

print(f"\nPermutation test done.")
print(f"  Observed difference = {observed_diff:.4f}")
print(f"  Mean(perm_diffs)    = {perm_diffs.mean():.4f}")
print(f"  Two-sided p-value   = {p_value:.4f}")

reversed_perm_diffs = -perm_diffs
reversed_observed_diff = -observed_diff
plt.figure(figsize=(8, 5))
plt.hist(reversed_perm_diffs, bins=20, alpha=0.7)
plt.axvline(reversed_observed_diff, linestyle='--', color="red")
plt.xlabel("Redundancy Score")
plt.ylabel("Count")
plt.title("")
plt.tight_layout()
plt.show()
