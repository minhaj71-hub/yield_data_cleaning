##############Dataset - Raw Data##################
# Import required Libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import pickle
import time
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Set the working directory
os.chdir("working_directory/Outputs/")

# Set output directory
output_dir = "Model Results/CB/Raw/"
os.makedirs(output_dir, exist_ok=True)

# Load data (Loaded the csv file created by the Data Preprocessing)
df = pd.read_csv("combined_data.csv")

df["near_edge"] = df["near_edge"].replace({"near_edge": 1, "not_near_edge": 0})

# Remove duplicates
df.drop_duplicates(inplace=True)
df.dropna(subset=["RmCode"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract features and labels
X = df.drop("RmCode", axis=1)
y = df["RmCode"]

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

######### Start - Bayesian Optimization ###########
param_space = {
    'bagging_temperature': Real(0.0, 1.0),
    'border_count': Integer(32, 255),
    'colsample_bylevel': Real(0.5, 1.0),
    'depth': Integer(4, 10),
    'iterations': Integer(100, 1000),
    'l2_leaf_reg': Real(1e-3, 10, prior='log-uniform'),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
}

# Initialize CatBoostClassifier
cb = CatBoostClassifier(verbose=0,  random_state=42, thread_count=-1)

# Set up Bayesian Optimization
bayes_search = BayesSearchCV(
    estimator=cb,
    search_spaces=param_space,
    n_iter=32,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

bayes_search.fit(X_train, y_train)

# Extract and save results
cv_results_bayes = bayes_search.cv_results_
results_df = pd.DataFrame(cv_results_bayes)
results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'rank_test_score']]
results_df.to_csv(output_dir + 'catboost_bayesian_optimization_results.csv', index=False)

best_params = bayes_search.best_params_
best_score = bayes_search.best_score_

best_results = {
    'Parameter': list(best_params.keys()) + ['Best F1 Score'],
    'Value': list(best_params.values()) + [best_score]
}
best_results_df = pd.DataFrame(best_results)
best_results_df.to_csv(output_dir + 'catboost_best_parameters_and_score.csv', index=False)

print("Best parameters found for CatBoost: ", best_params)
print("Best F1 Score for CatBoost: ", best_score)
print("Best parameters and F1 score saved to 'catboost_best_parameters_and_score.csv'")
######### End - Bayesian Optimization ###########


# Initialize CatBoost model
best_cb_model_full = CatBoostClassifier(**best_params, verbose=0, thread_count=-1, random_state = 42)

best_cb_model_full.fit(X_train, y_train)

############### Start - Cross-validation for all the metrics ################
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": "roc_auc"
}
cv_results = cross_validate(
    best_cb_model_full,
    X_train,
    y_train,
    cv=5,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)
pd.DataFrame(cv_results).to_csv(output_dir + "catboost_cross_val_results.csv", index=False)
############### End - Cross-validation for all the metrics ################


######## Start - Out of sample Prediction and ROC curve plotting ###########
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out_of_sample_preds = np.zeros(len(X_train))
plt.figure(figsize=(10, 8))

roc_data_summary = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Re-initialize model for each fold
    fold_model = CatBoostClassifier(**best_params, verbose=0, thread_count = -1, random_state = 42)
    fold_model.fit(X_train_fold, y_train_fold)

    # Predict probabilities
    y_val_probs = fold_model.predict_proba(X_val_fold)[:, 1]
    out_of_sample_preds[val_index] = y_val_probs

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_probs)
    fold_roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {fold_roc_auc:.2f})")

    # Store summary
    roc_data_summary.append({"Fold": fold, "AUC": fold_roc_auc})

# Random-chance line
plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Chance")

plt.title("ROC Curves for Cross-Validation Folds (CatBoost)", fontsize=16)
plt.xlabel("Specificity (False Positive Rate)", fontsize=14)
plt.ylabel("Sensitivity (True Positive Rate)", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

plt.savefig(output_dir + "cv_roc_curves.png", dpi=300)
plt.show()

pd.DataFrame(roc_data_summary).to_csv(output_dir + "roc_summary.csv", index=False)
pd.DataFrame(out_of_sample_preds, columns=["CatBoost_Preds"]).to_csv(output_dir + "oof_preds.csv", index=False)
######## End - Out of sample Prediction and ROC curve plotting ###########

######## Start - Train base learner #########
start_time = time.time()
best_cb_model_full.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
pd.DataFrame({"Metric": ["Training Time (seconds)"], "Score": [training_time]}).to_csv(
    output_dir + "training_time.csv", index=False
)
print(f"Total training time was {training_time:.2f} seconds")

# Save the final trained model
with open(output_dir + "model_full.pkl", "wb") as f:
    pickle.dump(best_cb_model_full, f)
######## End - Train base learner #########

######## Start - Evaluation on train dataset ##########
y_train_pred_best_full = best_cb_model_full.predict(X_train)
y_train_probs_best_full = best_cb_model_full.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred_best_full)
train_precision = precision_score(y_train, y_train_pred_best_full)
train_recall = recall_score(y_train, y_train_pred_best_full)
train_f1 = f1_score(y_train, y_train_pred_best_full)
train_roc_auc = roc_auc_score(y_train, y_train_probs_best_full)

print("\nTraining Set Results:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print(f"ROC-AUC Score: {train_roc_auc:.4f}")
######## End - Evaluation on train dataset ##########

######## Start - Evaluation on test dataset  ##########
y_test_pred_best_full = best_cb_model_full.predict(X_test)
y_test_probs_best_full = best_cb_model_full.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred_best_full)
test_precision = precision_score(y_test, y_test_pred_best_full)
test_recall = recall_score(y_test, y_test_pred_best_full)
test_f1 = f1_score(y_test, y_test_pred_best_full)
test_roc_auc = roc_auc_score(y_test, y_test_probs_best_full)
test_conf_matrix = confusion_matrix(y_test, y_test_pred_best_full)

print("\nTest Set Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC Score: {test_roc_auc:.4f}")

# Save metrics
eval_results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    "Training Score": [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc],
    "Test Score": [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc],
}
pd.DataFrame(eval_results).to_csv(output_dir + "evaluation_matrix.csv", index=False)

print("\nTraining and Test metrics saved to 'evaluation_matrix.csv'")

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(test_conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Best CatBoost (Full Dataset)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Non-Erroneous", "Erroneous"], rotation=45)
plt.yticks(tick_marks, ["Non-Erroneous", "Erroneous"])

thresh_best_full = test_conf_matrix.max() / 2.0
for i in range(test_conf_matrix.shape[0]):
    for j in range(test_conf_matrix.shape[1]):
        plt.text(
            j,
            i,
            format(test_conf_matrix[i, j], ".2f"),
            horizontalalignment="center",
            color="white" if test_conf_matrix[i, j] > thresh_best_full else "black"
        )

plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(output_dir + "confusion_matrix.png")
plt.show()
######### End - Evaluation on test dataset ############  










#############Dataset - Raw data with Weight Balancing#################
best_cb_model_full = CatBoostClassifier(**best_params, verbose=0, thread_count=-1, random_state = 42)

# Set the working directory
output_dir = "Model Results/CB/Raw+wb/"
os.makedirs(output_dir, exist_ok=True)


# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Convert to sample weights
sample_weights = np.array([class_weight_dict[label] for label in y_train])


########### Start - Manual cross validation #############
# Define the scoring metrics
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store CV results
cv_results = {metric: [] for metric in scoring_metrics}

# Perform manual cross-validation
for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Compute sample weights for the fold
    sample_weights_fold = np.array([class_weight_dict[label] for label in y_train_fold])
    
    # Re-initialize model for each fold
    model = CatBoostClassifier(**best_params, verbose=0, thread_count = -1, random_state = 42)
    model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)

    # Predictions
    y_val_pred = model.predict(X_val_fold)
    y_val_probs = model.predict_proba(X_val_fold)[:, 1]  # For ROC-AUC

    # Store scores
    cv_results['accuracy'].append(accuracy_score(y_val_fold, y_val_pred))
    cv_results['precision'].append(precision_score(y_val_fold, y_val_pred))
    cv_results['recall'].append(recall_score(y_val_fold, y_val_pred))
    cv_results['f1'].append(f1_score(y_val_fold, y_val_pred))
    cv_results['roc_auc'].append(roc_auc_score(y_val_fold, y_val_probs))

# Print averaged cross-validation results
print("\nCross-Validation Results:")
for metric in cv_results:
    print(f"{metric.capitalize()}: {np.mean(cv_results[metric]):.4f} (± {np.std(cv_results[metric]):.4f})")



# Extract the results and save them to a CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(output_dir + 'cross_val_results.csv', index=False)

########### End - Manual cross validation #############


########### Start - Out of sample Prediction #############
out_of_sample_preds = np.zeros(len(X_train))
true_labels = np.zeros(len(X_train))  # To store true labels for all folds
plt.figure(figsize=(10, 8))  # Initialize the ROC plot

roc_data_summary = []  # To store the summary (Fold and AUC)

# Generate out-of-sample predictions and plot ROC for each fold
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    sample_weights_fold = np.array([class_weight_dict[label] for label in y_train_fold])
    
    fold_model = CatBoostClassifier(**best_params, verbose=0, thread_count=-1, random_state=42)
    fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)

    
    # Predict probabilities on the validation fold
    y_val_probs = fold_model.predict_proba(X_val_fold)[:, 1]
    out_of_sample_preds[val_index] = y_val_probs
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve for this fold
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.2f})")
    
    # Append only the summary result (Fold and AUC) to the summary list
    roc_data_summary.append({'Fold': fold, 'AUC': roc_auc})

# Plot the diagonal line (random chance)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')

# Customize the plot
plt.title("ROC Curves for Cross-Validation Folds", fontsize=16)
plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=14)
plt.ylabel("Sensitivity (True Positive Rate)", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

# Save the plot as an image
plt.savefig(output_dir + 'cv_roc_curves.png', dpi=300)
plt.show()

# Save only the summary results (Fold and AUC) to CSV
pd.DataFrame(roc_data_summary).to_csv(output_dir + 'roc_summary.csv', index=False)

# Save out-of-sample predictions to CSV
pd.DataFrame(out_of_sample_preds, columns=["CB_Preds"]).to_csv(output_dir + "oof_preds.csv", index=False)
########### END - Out of sample Prediction #############

########### Start - Train Base Learner #############
start_time = time.time()
best_cb_model_full.fit(X_train, y_train, sample_weight=sample_weights)
end_time = time.time()

training_time = end_time - start_time
print(f"Total training time was {training_time: .2f} seconds")

training_time_df = pd.DataFrame({'Metric': ['Training Time (seconds)'], 'Score' : [training_time]})
training_time_df.to_csv(output_dir + 'training_time.csv', index=False)

# Save the final trained model
with open(output_dir + 'model_full.pkl', 'wb') as f:
    pickle.dump(best_cb_model_full, f)
########### End - Train Base Learner #############    


# ========= EVALUATE THE MODEL ON TRAINING DATA =========
y_train_pred_best_full = best_cb_model_full.predict(X_train)
y_train_probs_best_full = best_cb_model_full.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred_best_full)
train_precision = precision_score(y_train, y_train_pred_best_full)
train_recall = recall_score(y_train, y_train_pred_best_full)
train_f1 = f1_score(y_train, y_train_pred_best_full)
train_roc_auc = roc_auc_score(y_train, y_train_probs_best_full)

# Print Training Metrics
print("\nTraining Set Results:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print(f"ROC-AUC Score: {train_roc_auc:.4f}")

# ========= EVALUATE THE MODEL ON TESTING DATA =========
y_test_pred_best_full = best_cb_model_full.predict(X_test)

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_pred_best_full)
test_precision = precision_score(y_test, y_test_pred_best_full)
test_recall = recall_score(y_test, y_test_pred_best_full)
test_f1 = f1_score(y_test, y_test_pred_best_full)
test_roc_auc = roc_auc_score(y_test, best_cb_model_full.predict_proba(X_test)[:, 1])
test_conf_matrix = confusion_matrix(y_test, y_test_pred_best_full)

# Print metrics
print("Test Set Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC Score: {test_roc_auc:.4f}")

# Save both training and test evaluation metrics
eval_results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'Training Score': [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc],
    'Test Score': [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc]
}

eval_results_df = pd.DataFrame(eval_results)
eval_results_df.to_csv(output_dir + 'evaluation_matrix.csv', index=False)

print("\nTraining and Test metrics saved to 'evaluation_matrix.csv'")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(test_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Best Model (Full Dataset)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Erroneous', 'Erroneous'], rotation=45)
plt.yticks(tick_marks, ['Non-Erroneous', 'Erroneous'])

thresh_best_full = test_conf_matrix.max() / 2.
for i, j in np.ndindex(test_conf_matrix.shape):
    plt.text(j, i, format(test_conf_matrix[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if test_conf_matrix[i, j] > thresh_best_full else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(output_dir +"confusion_matrix.png")
plt.show()












##############Dataset - SMOTE Data##################
best_cb_model_full = CatBoostClassifier(**best_params, verbose=0, thread_count=-1, random_state = 42)

# Set output directory
output_dir = "Model Results/CB/SMOTE/"
os.makedirs(output_dir, exist_ok=True)

# Step 2: Apply SMOTE on the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

####### Start - Cross validation ########
cv_results = cross_validate(best_cb_model_full, X_train_smote, y_train_smote, cv=5, scoring=scoring, return_train_score=False, n_jobs=-1)

# Display the cross-validation results
print("Cross-validation results:")
for metric in scoring:
    print(f"{metric.capitalize()}: {cv_results['test_' + metric].mean():.4f} (± {cv_results['test_' + metric].std():.4f})")

# Extract the results and save them to a CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(output_dir + 'cross_val_results.csv', index=False)
####### End - Cross validation ########


####### Start - Out of Sample Prediction #########
out_of_sample_preds = np.zeros(len(X_train_smote))
true_labels = np.zeros(len(X_train_smote))  # To store true labels for all folds
plt.figure(figsize=(10, 8))  # Initialize the ROC plot

roc_data_summary = []  # To store the summary (Fold and AUC)

# Generate out-of-sample predictions and plot ROC for each fold
for fold, (train_index, val_index) in enumerate(skf.split(X_train_smote, y_train_smote), 1):
    X_train_fold, X_val_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[val_index]
    y_train_fold, y_val_fold = y_train_smote.iloc[train_index], y_train_smote.iloc[val_index]
    
    fold_model = CatBoostClassifier(**best_params, verbose=0, thread_count=-1, random_state=42)
    fold_model.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities on the validation fold
    y_val_probs = fold_model.predict_proba(X_val_fold)[:, 1]
    out_of_sample_preds[val_index] = y_val_probs
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve for this fold
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.2f})")
    
    # Append only the summary result (Fold and AUC) to the summary list
    roc_data_summary.append({'Fold': fold, 'AUC': roc_auc})

# Plot the diagonal line (random chance)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')

# Customize the plot
plt.title("ROC Curves for Cross-Validation Folds", fontsize=16)
plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=14)
plt.ylabel("Sensitivity (True Positive Rate)", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

# Save the plot as an image
plt.savefig(output_dir + 'cv_roc_curves.png', dpi=300)
plt.show()

# Save only the summary results (Fold and AUC) to CSV
pd.DataFrame(roc_data_summary).to_csv(output_dir + 'roc_summary.csv', index=False)

# Save out-of-sample predictions to CSV
pd.DataFrame(out_of_sample_preds, columns=["CB_Preds"]).to_csv(output_dir + "oof_preds.csv", index=False)
####### End - Out of Sample Prediction #########

####### Start - Train base Learner #########
start_time = time.time()
best_cb_model_full.fit(X_train_smote, y_train_smote)
end_time = time.time()

training_time = end_time - start_time
print(f"Total training time was {training_time: .2f} seconds")

training_time_df = pd.DataFrame({'Metric': ['Training Time (seconds)'], 'Score' : [training_time]})
training_time_df.to_csv(output_dir + 'training_time.csv', index=False)

# Save the final trained model
with open(output_dir + 'model_full.pkl', 'wb') as f:
    pickle.dump(best_cb_model_full, f)
####### End - Train base Learner #########

# ========= EVALUATE THE MODEL ON TRAINING DATA =========
y_train_pred_best_full = best_cb_model_full.predict(X_train_smote)
y_train_probs_best_full = best_cb_model_full.predict_proba(X_train_smote)[:, 1]

train_accuracy = accuracy_score(y_train_smote, y_train_pred_best_full)
train_precision = precision_score(y_train_smote, y_train_pred_best_full)
train_recall = recall_score(y_train_smote, y_train_pred_best_full)
train_f1 = f1_score(y_train_smote, y_train_pred_best_full)
train_roc_auc = roc_auc_score(y_train_smote, y_train_probs_best_full)

# Print Training Metrics
print("\nTraining Set Results:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print(f"ROC-AUC Score: {train_roc_auc:.4f}")

# ========= EVALUATE THE MODEL ON TESTING DATA =========
y_test_pred_best_full = best_cb_model_full.predict(X_test)

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_pred_best_full)
test_precision = precision_score(y_test, y_test_pred_best_full)
test_recall = recall_score(y_test, y_test_pred_best_full)
test_f1 = f1_score(y_test, y_test_pred_best_full)
test_roc_auc = roc_auc_score(y_test, best_cb_model_full.predict_proba(X_test)[:, 1])
test_conf_matrix = confusion_matrix(y_test, y_test_pred_best_full)

# Print metrics
print("Test Set Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC Score: {test_roc_auc:.4f}")

# Save both training and test evaluation metrics
eval_results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'Training Score': [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc],
    'Test Score': [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc]
}

eval_results_df = pd.DataFrame(eval_results)
eval_results_df.to_csv(output_dir + 'evaluation_matrix.csv', index=False)

print("\nTraining and Test metrics saved to 'evaluation_matrix.csv'")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(test_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Best Model (Full Dataset)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Erroneous', 'Erroneous'], rotation=45)
plt.yticks(tick_marks, ['Non-Erroneous', 'Erroneous'])

thresh_best_full = test_conf_matrix.max() / 2.
for i, j in np.ndindex(test_conf_matrix.shape):
    plt.text(j, i, format(test_conf_matrix[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if test_conf_matrix[i, j] > thresh_best_full else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(output_dir +"confusion_matrix.png")
plt.show()











##############Dataset - NearMiss-1 Data##################
# Set output directory
output_dir = "Model Results/CB/NearMiss/"
os.makedirs(output_dir, exist_ok=True)

# Apply NearMiss (Version 1) for undersampling
nearmiss = NearMiss(version=1)
X_nearmiss, y_nearmiss = nearmiss.fit_resample(X_train, y_train)

######### Start - Cross validation ##########
cv_results = cross_validate(best_cb_model_full, X_nearmiss, y_nearmiss, cv=5, scoring=scoring, return_train_score=False, n_jobs=-1)

# Display the cross-validation results
print("Cross-validation results:")
for metric in scoring:
    print(f"{metric.capitalize()}: {cv_results['test_' + metric].mean():.4f} (± {cv_results['test_' + metric].std():.4f})")

# Extract the results and save them to a CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(output_dir + 'cross_val_results.csv', index=False)
######### End - Cross validation ##########


######### Start - Out of Sample Prediction ###########
out_of_sample_preds = np.zeros(len(X_nearmiss))
true_labels = np.zeros(len(X_nearmiss))  # To store true labels for all folds
plt.figure(figsize=(10, 8))  # Initialize the ROC plot

roc_data_summary = []  # To store the summary (Fold and AUC)

# Generate out-of-sample predictions and plot ROC for each fold
for fold, (train_index, val_index) in enumerate(skf.split(X_nearmiss, y_nearmiss), 1):
    X_train_fold, X_val_fold = X_nearmiss.iloc[train_index], X_nearmiss.iloc[val_index]
    y_train_fold, y_val_fold = y_nearmiss.iloc[train_index], y_nearmiss.iloc[val_index]
    
    fold_model = CatBoostClassifier(**best_params, verbose=0, thread_count=-1, random_state=42)
    fold_model.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities on the validation fold
    y_val_probs = best_cb_model_full.predict_proba(X_val_fold)[:, 1]
    out_of_sample_preds[val_index] = y_val_probs  # Store probabilities for OOF
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve for this fold
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.2f})")
    
    # Append only the summary result (Fold and AUC) to the summary list
    roc_data_summary.append({'Fold': fold, 'AUC': roc_auc})

# Plot the diagonal line (random chance)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')

# Customize the plot
plt.title("ROC Curves for Cross-Validation Folds", fontsize=16)
plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=14)
plt.ylabel("Sensitivity (True Positive Rate)", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

# Save the plot as an image
plt.savefig(output_dir + 'cv_roc_curves.png', dpi=300)
plt.show()

# Save only the summary results (Fold and AUC) to CSV
pd.DataFrame(roc_data_summary).to_csv(output_dir + 'roc_summary.csv', index=False)

# Save out-of-sample predictions to CSV
pd.DataFrame(out_of_sample_preds, columns=["CB_Preds"]).to_csv(output_dir + "oof_preds.csv", index=False)
######### End - Out of sample prediction ###########

######### Start - Train base learner ###########
start_time = time.time()
best_cb_model_full.fit(X_nearmiss, y_nearmiss)
end_time = time.time()

training_time = end_time - start_time
print(f"Total training time was {training_time: .2f} seconds")

training_time_df = pd.DataFrame({'Metric': ['Training Time (seconds)'], 'Score' : [training_time]})
training_time_df.to_csv(output_dir + 'training_time.csv', index=False)

# Save the final trained model
with open(output_dir + 'model_full.pkl', 'wb') as f:
    pickle.dump(best_cb_model_full, f)
######### End - Train base learner ###########

# ========= EVALUATE THE MODEL ON TRAINING DATA =========
y_train_pred_best_full = best_cb_model_full.predict(X_nearmiss)
y_train_probs_best_full = best_cb_model_full.predict_proba(X_nearmiss)[:, 1]

train_accuracy = accuracy_score(y_nearmiss, y_train_pred_best_full)
train_precision = precision_score(y_nearmiss, y_train_pred_best_full)
train_recall = recall_score(y_nearmiss, y_train_pred_best_full)
train_f1 = f1_score(y_nearmiss, y_train_pred_best_full)
train_roc_auc = roc_auc_score(y_nearmiss, y_train_probs_best_full)

# Print Training Metrics
print("\nTraining Set Results:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print(f"ROC-AUC Score: {train_roc_auc:.4f}")

# ========= EVALUATE THE MODEL ON TESTING DATA =========
y_test_pred_best_full = best_cb_model_full.predict(X_test)

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_pred_best_full)
test_precision = precision_score(y_test, y_test_pred_best_full)
test_recall = recall_score(y_test, y_test_pred_best_full)
test_f1 = f1_score(y_test, y_test_pred_best_full)
test_roc_auc = roc_auc_score(y_test, best_cb_model_full.predict_proba(X_test)[:, 1])
test_conf_matrix = confusion_matrix(y_test, y_test_pred_best_full)

# Print metrics
print("Test Set Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC Score: {test_roc_auc:.4f}")

# Save both training and test evaluation metrics
eval_results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'Training Score': [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc],
    'Test Score': [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc]
}

eval_results_df = pd.DataFrame(eval_results)
eval_results_df.to_csv(output_dir + 'evaluation_matrix.csv', index=False)

print("\nTraining and Test metrics saved to 'evaluation_matrix.csv'")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(test_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Best Model (Full Dataset)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Erroneous', 'Erroneous'], rotation=45)
plt.yticks(tick_marks, ['Non-Erroneous', 'Erroneous'])

thresh_best_full = test_conf_matrix.max() / 2.
for i, j in np.ndindex(test_conf_matrix.shape):
    plt.text(j, i, format(test_conf_matrix[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if test_conf_matrix[i, j] > thresh_best_full else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(output_dir +"confusion_matrix.png")
plt.show()