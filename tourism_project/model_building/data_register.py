# Registers the raw tourism dataset on the Hugging Face Hub as a dataset repo
import os
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "MatthewSimon/tourism-dataset"   # change to your username
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repo only if it doesn't already exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the local data folder to the dataset repo
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded successfully.")
