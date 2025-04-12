import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_csv_path = "/content/drive/MyDrive/num_train.csv"
test_csv_path = "/content/drive/MyDrive/num_test.csv"
pairs = [(2, 4), (3, 14), (6, 7), (8, 9)]

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

print("Loading and transforming training data...")
train_power = load_and_transform(train_csv_path)
print("Loading and transforming test data...")
test_power = load_and_transform(test_csv_path)
print(f"train_power shape: {train_power.shape}")
print(f"test_power shape:  {test_power.shape}")

y_train = np.array([1] * 120 + [0] * 120)
y_test = np.array([1] * 30 + [0] * 30)

if train_power.shape[0] != 240:
    raise ValueError(f"Expected 240 training samples, got {train_power.shape[0]}")
if test_power.shape[0] != 60:
    raise ValueError(f"Expected 60 test samples, got {test_power.shape[0]}")

def get_features_for_pair(power_data, row_pair, method='separate'):
    row1, row2 = row_pair
    row1 -= 1
    row2 -= 1
    subset = power_data[:, [row1, row2], :]
    if method == 'separate':
        return subset.reshape(subset.shape[0], -1)
    elif method == 'difference':
        return subset[:, 0, :] - subset[:, 1, :]
    elif method == 'average':
        return (subset[:, 0, :] + subset[:, 1, :]) / 2.0
    else:
        raise ValueError("method must be 'separate', 'difference', or 'average'")

methods = ["separate", "difference", "average"]
results = {"pair_names": [], "acc_sep": [], "acc_diff": [], "acc_avg": []}

print("\nComparing (Separate vs. Difference vs. Average) for each of the 4 pairs...\n")
for pair in pairs:
    X_train_sep = get_features_for_pair(train_power, pair, 'separate')
    X_test_sep = get_features_for_pair(test_power, pair, 'separate')
    X_train_diff = get_features_for_pair(train_power, pair, 'difference')
    X_test_diff = get_features_for_pair(test_power, pair, 'difference')
    X_train_avg = get_features_for_pair(train_power, pair, 'average')
    X_test_avg = get_features_for_pair(test_power, pair, 'average')
    
    scaler_sep = StandardScaler()
    X_train_sep_scaled = scaler_sep.fit_transform(X_train_sep)
    X_test_sep_scaled = scaler_sep.transform(X_test_sep)
    
    scaler_diff = StandardScaler()
    X_train_diff_scaled = scaler_diff.fit_transform(X_train_diff)
    X_test_diff_scaled = scaler_diff.transform(X_test_diff)
    
    scaler_avg = StandardScaler()
    X_train_avg_scaled = scaler_avg.fit_transform(X_train_avg)
    X_test_avg_scaled = scaler_avg.transform(X_test_avg)
    
    lr_sep = LogisticRegression(max_iter=1000)
    lr_sep.fit(X_train_sep_scaled, y_train)
    acc_sep = accuracy_score(y_test, lr_sep.predict(X_test_sep_scaled))
    
    lr_diff = LogisticRegression(max_iter=1000)
    lr_diff.fit(X_train_diff_scaled, y_train)
    acc_diff = accuracy_score(y_test, lr_diff.predict(X_test_diff_scaled))
    
    lr_avg = LogisticRegression(max_iter=1000)
    lr_avg.fit(X_train_avg_scaled, y_train)
    acc_avg = accuracy_score(y_test, lr_avg.predict(X_test_avg_scaled))
    
    pair_str = f"({pair[0]},{pair[1]})"
    print(f"Pair {pair_str}:")
    print(f"  Separate Acc:   {acc_sep:.4f}")
    print(f"  Difference Acc: {acc_diff:.4f}")
    print(f"  Average Acc:    {acc_avg:.4f}\n")
    
    results["pair_names"].append(pair_str)
    results["acc_sep"].append(acc_sep)
    results["acc_diff"].append(acc_diff)
    results["acc_avg"].append(acc_avg)

pair_count = len(pairs)
x_indices = np.arange(pair_count)
width = 0.25

plt.figure(figsize=(10, 5))
ax = plt.gca()
bars_sep = ax.bar(x_indices - width, results["acc_sep"], width, label='Separate')
bars_diff = ax.bar(x_indices, results["acc_diff"], width, label='Difference')
bars_avg = ax.bar(x_indices + width, results["acc_avg"], width, label='Average')
ax.set_title("Comparison of Separate vs. Difference vs. Average\n(One Pair at a Time)")
ax.set_xlabel("Row Pair (1-based indices)")
ax.set_ylabel("Test Accuracy")
ax.set_xticks(x_indices)
ax.set_xticklabels(results["pair_names"], rotation=0)
ax.set_ylim([0, 1])
ax.legend()

def add_labels(bar_container):
    for bar in bar_container:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h+0.01, f"{h:.2f}", ha='center', va='bottom', fontsize=8)

add_labels(bars_sep)
add_labels(bars_diff)
add_labels(bars_avg)
plt.tight_layout()
plt.show()
