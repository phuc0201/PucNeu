from pathlib import Path

from huggingface_hub import snapshot_download


MODELS_DIR = Path(__file__).resolve().parent / "models"

REPOSITORIES = {
    "pnnbao-ump/VieNeu-TTS": "VieNeu-TTS",
    "neuphonic/neucodec": "neucodec",
    "neuphonic/distill-neucodec": "distill-neucodec",
}

for repo_id, target_dir_name in REPOSITORIES.items():
    local_dir = MODELS_DIR / target_dir_name
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )