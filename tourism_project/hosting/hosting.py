# Hosting: push the deployment files (Dockerfile, app.py, requirements.txt)
# to a Hugging Face Space, which builds and serves the Streamlit app
import os
from huggingface_hub import HfApi, create_repo

SPACE_REPO = "MatthewSimon/tourism-wellness-app"
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

# Create the Space if it doesn't exist; space_sdk="docker" tells HF to build
# from our Dockerfile. exist_ok=True makes reruns safe.
create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
    token=token,
)

# Upload the deployment folder to the root of the Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files pushed to https://huggingface.co/spaces/{SPACE_REPO}")
