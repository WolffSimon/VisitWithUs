# Data preparation: load raw data from the HF dataset repo, clean it,
# split into train/test, save locally, and upload the splits back to HF
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

repo_id = "MatthewSimon/tourism-dataset"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the raw dataset directly from the Hugging Face dataset repo
df = pd.read_csv(f"hf://datasets/{repo_id}/tourism.csv")
print("Raw data loaded. Shape:", df.shape)

# --- Cleaning (decisions justified in the EDA section of the notebook) ---
# 1. Drop identifier columns (leftover index and unique customer ID), and
#    ProductPitched, which maps one-to-one onto Designation (see EDA) and so
#    adds no information while splitting feature importance
df = df.drop(columns=["Unnamed: 0", "CustomerID", "ProductPitched"])

# 2. Correct the data entry error in Gender
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# 3. Remove exact duplicate rows (same customer entered under different IDs)
df = df.drop_duplicates()

# 'Unmarried' is kept as a distinct MaritalStatus category, and numeric
# outliers are retained, since the tree-based models are robust to them
print("Cleaned data. Shape:", df.shape)

# --- Train/test split ---
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratify on the target so both sets keep the same share of buyers
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits locally
out_dir = "tourism_project/data"
files = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest,
         "ytrain.csv": ytrain, "ytest.csv": ytest}
for name, frame in files.items():
    frame.to_csv(os.path.join(out_dir, name), index=False)

# Upload each split back to the HF dataset repo
for name in files:
    api.upload_file(
        path_or_fileobj=os.path.join(out_dir, name),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {name}")
