# Downloads the entirety of the 5 countries DHS dataset.
# Use only once before further data extraction.

import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mariaamandadevina/dhs-dataset",
    repo_type="dataset",
    local_dir="__data__/",
    token=os.getenv("HF_TOKEN"),
)
