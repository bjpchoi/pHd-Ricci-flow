"""
Sample starter script.

"""

import numpy as np
from math import log, sqrt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


param_grid = {
   'pca__n_components': [10, 30, 50],
   'svm__C': [100, 1000, 5000],
   'svm__gamma': [1e-4, 1e-3],
   'svm__kernel': ['rbf'] # small sample grid → add more as needed
}


n_runs = 10


test_accs_mod = []
test_accs_unmod = []
test_accs_dasm = []


train_accs_mod = []
train_accs_unmod = []
train_accs_dasm = []


log_losses_mod = []
log_losses_unmod = []
log_losses_dasm = []


for run_idx in range(n_runs):
   print(f"\n========== Shuffle Run {run_idx+1} / {n_runs} ==========")


   skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run_idx)


   pipeline_mod = Pipeline([
       ('pca', PCA()),
       ('svm', SVC(probability=True))
   ])


   grid_search_mod = GridSearchCV(
       estimator=pipeline_mod,
       param_grid=param_grid,
       cv=skf,
       scoring='accuracy',
       n_jobs=-1,
       verbose=0  
   )


   grid_search_mod.fit(X_train_mod_scaled, y_train)
  
   best_model_mod = grid_search_mod.best_estimator_
   train_preds_mod = best_model_mod.predict(X_train_mod_scaled)
   test_preds_mod = best_model_mod.predict(X_test_mod_scaled)


   train_acc_mod = accuracy_score(y_train, train_preds_mod)
   test_acc_mod = accuracy_score(y_test, test_preds_mod)
  
   print("\n--- Modified Feature Set ---")
   print("Best Params:", grid_search_mod.best_params_)
   print(f"Train Accuracy: {train_acc_mod:.4f}, Test Accuracy: {test_acc_mod:.4f}")
  
   train_accs_mod.append(train_acc_mod)
   test_accs_mod.append(test_acc_mod)
  
   y_proba_mod = best_model_mod.predict_proba(X_test_mod_scaled)
   individual_log_losses_mod = []
   for i, true_label in enumerate(y_test):
       p_correct = max(y_proba_mod[i, true_label], 1e-15)  # clip prob
       loss_i = -log(p_correct)
       individual_log_losses_mod.append(loss_i)
   log_losses_mod.append(np.mean(individual_log_losses_mod))


   pipeline_unmod = Pipeline([
       ('pca', PCA()),
       ('svm', SVC(probability=True))
   ])


   grid_search_unmod = GridSearchCV(
       estimator=pipeline_unmod,
       param_grid=param_grid,
       cv=skf,
       scoring='accuracy',
       n_jobs=-1,
       verbose=0
   )


   grid_search_unmod.fit(X_train_unmod_scaled, y_train)
  
   best_model_unmod = grid_search_unmod.best_estimator_
   train_preds_unmod = best_model_unmod.predict(X_train_unmod_scaled)
   test_preds_unmod = best_model_unmod.predict(X_test_unmod_scaled)


   train_acc_unmod = accuracy_score(y_train, train_preds_unmod)
   test_acc_unmod = accuracy_score(y_test, test_preds_unmod)
  
   print("\n--- Unmodified Feature Set ---")
   print("Best Params:", grid_search_unmod.best_params_)
   print(f"Train Accuracy: {train_acc_unmod:.4f}, Test Accuracy: {test_acc_unmod:.4f}")
  
   train_accs_unmod.append(train_acc_unmod)
   test_accs_unmod.append(test_acc_unmod)


   y_proba_unmod = best_model_unmod.predict_proba(X_test_unmod_scaled)
   individual_log_losses_unmod = []
   for i, true_label in enumerate(y_test):
       p_correct = max(y_proba_unmod[i, true_label], 1e-15)
       loss_i = -log(p_correct)
       individual_log_losses_unmod.append(loss_i)
   log_losses_unmod.append(np.mean(individual_log_losses_unmod))


   pipeline_dasm = Pipeline([
       ('pca', PCA()),
       ('svm', SVC(probability=True))
   ])


   grid_search_dasm = GridSearchCV(
       estimator=pipeline_dasm,
       param_grid=param_grid,
       cv=skf,
       scoring='accuracy',
       n_jobs=-1,
       verbose=0
   )


   grid_search_dasm.fit(X_train_dasm_rasm_de_scaled, y_train)
  
   best_model_dasm = grid_search_dasm.best_estimator_
   train_preds_dasm = best_model_dasm.predict(X_train_dasm_rasm_de_scaled)
   test_preds_dasm = best_model_dasm.predict(X_test_dasm_rasm_de_scaled)


   train_acc_dasm = accuracy_score(y_train, train_preds_dasm)
   test_acc_dasm = accuracy_score(y_test, test_preds_dasm)
  
   print("\n--- DE+DASM+RASM Feature Set ---")
   print("Best Params:", grid_search_dasm.best_params_)
   print(f"Train Accuracy: {train_acc_dasm:.4f}, Test Accuracy: {test_acc_dasm:.4f}")
  
   train_accs_dasm.append(train_acc_dasm)
   test_accs_dasm.append(test_acc_dasm)
  
   y_proba_dasm = best_model_dasm.predict_proba(X_test_dasm_rasm_de_scaled)
   individual_log_losses_dasm = []
   for i, true_label in enumerate(y_test):
       p_correct = max(y_proba_dasm[i, true_label], 1e-15)
       loss_i = -log(p_correct)
       individual_log_losses_dasm.append(loss_i)
   log_losses_dasm.append(np.mean(individual_log_losses_dasm))


def mean_sem(values):
   arr = np.array(values)
   mean_ = np.mean(arr)
   std_ = np.std(arr, ddof=1)
   sem_ = std_ / sqrt(len(arr))
   return mean_, sem_


# Modified feature set
mean_test_mod, sem_test_mod = mean_sem(test_accs_mod)
print("\n============ Final Results: Modified Feature Set ============")
print(f"Test Accuracy across 10 runs: {mean_test_mod:.4f} ± {sem_test_mod:.4f}")


# Unmodified feature set
mean_test_unmod, sem_test_unmod = mean_sem(test_accs_unmod)
print("\n============ Final Results: Unmodified Feature Set ============")
print(f"Test Accuracy across 10 runs: {mean_test_unmod:.4f} ± {sem_test_unmod:.4f}")


# DE+DASM+RASM
mean_test_dasm, sem_test_dasm = mean_sem(test_accs_dasm)
print("\n============ Final Results: DE+DASM+RASM Feature Set ============")
print(f"Test Accuracy across 10 runs: {mean_test_dasm:.4f} ± {sem_test_dasm:.4f}")


