import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

diff_pair = ("CPZ", "O2")

print("Loading data...")
X_train = np.load(train_data_path)
X_test = np.load(test_data_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)
print("Data Loaded.")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

def get_time_domain_features_for_pair(X, channel_names, pair, method='separate'):
    ch_indices = [channel_names.index(ch) for ch in pair]
    subset = X[:, ch_indices, :]
    if method == 'separate':
        return subset.reshape(subset.shape[0], -1)
    elif method == 'difference':
        return subset[:, 0, :] - subset[:, 1, :]
    elif method == 'average':
        return subset.mean(axis=1)
    else:
        raise ValueError("method must be 'separate', 'difference', or 'average'")

X_train_sep = get_time_domain_features_for_pair(X_train, channel_names, diff_pair, 'separate')
X_test_sep = get_time_domain_features_for_pair(X_test, channel_names, diff_pair, 'separate')
X_train_diff = get_time_domain_features_for_pair(X_train, channel_names, diff_pair, 'difference')
X_test_diff = get_time_domain_features_for_pair(X_test, channel_names, diff_pair, 'difference')
X_train_avg = get_time_domain_features_for_pair(X_train, channel_names, diff_pair, 'average')
X_test_avg = get_time_domain_features_for_pair(X_test, channel_names, diff_pair, 'average')

lr_sep = LogisticRegression(max_iter=2000)
lr_sep.fit(X_train_sep, y_train)
acc_sep = accuracy_score(y_test, lr_sep.predict(X_test_sep))

lr_diff = LogisticRegression(max_iter=2000)
lr_diff.fit(X_train_diff, y_train)
acc_diff = accuracy_score(y_test, lr_diff.predict(X_test_diff))

lr_avg = LogisticRegression(max_iter=2000)
lr_avg.fit(X_train_avg, y_train)
acc_avg = accuracy_score(y_test, lr_avg.predict(X_test_avg))

print(f"\nDifference Pair {diff_pair} results (Raw Time Domain):")
print(f"  Separate channels accuracy:  {acc_sep:.4f}")
print(f"  Difference channels accuracy: {acc_diff:.4f}")
print(f"  Average channels accuracy:    {acc_avg:.4f}")

methods = ["Separate", "Difference", "Average"]
accs = [acc_sep, acc_diff, acc_avg]

plt.figure(figsize=(6, 4))
bars = plt.bar(methods, accs)
plt.ylim([0, 1])
plt.title(f"Accuracy Comparison for Pair {diff_pair}")
plt.ylabel("Test Accuracy")
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.4f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

print("Loading data...")
X_train = np.load(train_data_path)
X_test = np.load(test_data_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)
print("Data loaded.")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test: {y_test.shape}")

def get_time_domain_features_for_pair(X, channel_names, pair, method='separate'):
    ch_idx = [channel_names.index(ch) for ch in pair]
    subset = X[:, ch_idx, :]
    if method == 'separate':
        return subset.reshape(subset.shape[0], -1)
    elif method == 'difference':
        return subset[:, 0, :] - subset[:, 1, :]
    elif method == 'average':
        return subset.mean(axis=1)
    else:
        raise ValueError("method must be 'separate', 'difference', or 'average'")

methods_list = ["separate", "difference", "average"]
results_dict = {"pair_names": [], "acc_separate": [], "acc_difference": [], "acc_average": []}

print("\nEvaluating (Separate vs. Difference vs. Average) for each pair (Raw Time Domain)...\n")
for pair in avg_pairs:
    X_train_sep = get_time_domain_features_for_pair(X_train, channel_names, pair, 'separate')
    X_test_sep = get_time_domain_features_for_pair(X_test, channel_names, pair, 'separate')
    X_train_diff = get_time_domain_features_for_pair(X_train, channel_names, pair, 'difference')
    X_test_diff = get_time_domain_features_for_pair(X_test, channel_names, pair, 'difference')
    X_train_avg = get_time_domain_features_for_pair(X_train, channel_names, pair, 'average')
    X_test_avg = get_time_domain_features_for_pair(X_test, channel_names, pair, 'average')
    lr_sep = LogisticRegression(max_iter=2000)
    lr_sep.fit(X_train_sep, y_train)
    acc_sep = accuracy_score(y_test, lr_sep.predict(X_test_sep))
    lr_diff = LogisticRegression(max_iter=2000)
    lr_diff.fit(X_train_diff, y_train)
    acc_diff = accuracy_score(y_test, lr_diff.predict(X_test_diff))
    lr_avg = LogisticRegression(max_iter=2000)
    lr_avg.fit(X_train_avg, y_train)
    acc_avg = accuracy_score(y_test, lr_avg.predict(X_test_avg))
    pair_str = f"{pair[0]}-{pair[1]}"
    print(f"Pair {pair_str}:")
    print(f"  Separate Acc:  {acc_sep:.4f}")
    print(f"  Difference Acc: {acc_diff:.4f}")
    print(f"  Average Acc:    {acc_avg:.4f}\n")
    results_dict["pair_names"].append(pair_str)
    results_dict["acc_separate"].append(acc_sep)
    results_dict["acc_difference"].append(acc_diff)
    results_dict["acc_average"].append(acc_avg)

pairs_count = len(avg_pairs)
x = np.arange(pairs_count)
width = 0.25
fig, ax = plt.subplots(figsize=(12, 5))
bars_sep = ax.bar(x - width, results_dict["acc_separate"], width, label='Separate')
bars_diff = ax.bar(x, results_dict["acc_difference"], width, label='Difference')
bars_avg = ax.bar(x + width, results_dict["acc_average"], width, label='Average')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Separate vs. Difference vs. Average (Raw Time Domain)')
ax.set_xticks(x)
ax.set_xticklabels(results_dict["pair_names"], rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.legend()
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha='center', va='bottom', fontsize=8)
add_labels(bars_sep)
add_labels(bars_diff)
add_labels(bars_avg)
plt.tight_layout()
plt.show()
