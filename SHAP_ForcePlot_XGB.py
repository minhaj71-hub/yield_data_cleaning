##################Shap Analysis########################
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib
import matplotlib as mpl
import numpy as np

# Set the working directory
os.chdir("working_directory/Outputs/")

# Make text editable in SVG
mpl.rcParams['svg.fonttype'] = 'none'

#set output directory
output_dir = "Model Results/SHAP_Analysis/Raw_XGB/"
os.makedirs(output_dir, exist_ok=True)

model = joblib.load("Model Results/XGB/Raw/XGB_model_full.pkl")

# ====================================
# 1. Load and preprocess the dataset
# ====================================
df = pd.read_csv('combined_data.csv')

# Convert 'near_edge' to 0/1
df['near_edge'] = df['near_edge'].replace({'near_edge': 1, 'not_near_edge': 0})

# Remove duplicates and drop rows with missing RmCode
df.drop_duplicates(inplace=True)
df = df.dropna(subset=['RmCode'])
df.reset_index(drop=True, inplace=True)

# Extract features and labels
X = df.drop('RmCode', axis=1)
y = df['RmCode']

# Store feature names from the processed DataFrame
feature_names = X.columns.tolist()  # Extract feature names after preprocessing

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize the SHAP explainer for Gradient Boosting
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# Convert SHAP values to DataFrame for analysis
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)

# SHAP Summary Plot (Save as EPS)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(output_dir, "shap_summary.svg"), format='svg')
plt.savefig(os.path.join(output_dir, "shap_summary.png"), format='png', dpi=300)
plt.show()
print("SHAP summary plot saved as 'shap_summary.png'.")



##########Force Plots###########
n_samples = X_test.shape[0]

# Set random seed for reproducibility
np.random.seed(42)

# Randomly select 10 unique indices from the range of test set
selected_indices = np.random.choice(n_samples, size=10, replace=False)

# Create output directory
forceplot_dir = os.path.join(output_dir, "force_plots_svg")
os.makedirs(forceplot_dir, exist_ok=True)

# Update global plot settings
plt.rcParams.update({'font.size': 10})

# Generate and save force plots
for index in selected_indices:
    print(f"Generating force plot for index: {index}")

    shap_values_instance = shap_values[index]
    rounded_features = X_test.iloc[index].apply(lambda x: np.round(x, 2))

    # Generate force plot using SHAP's `matplotlib=True` mode
    plt.figure(figsize=(14, 5))
    shap.force_plot(
        base_value=np.round(shap_values_instance.base_values, 2),
        shap_values=np.round(shap_values_instance.values, 2),
        features=rounded_features,
        matplotlib=True
    )

    # Save plot in SVG format
    file_path = os.path.join(forceplot_dir, f"shap_force_plot_{index}.svg")
    plt.savefig(file_path, format='svg', bbox_inches='tight')
    plt.show()