from huggingface_hub import snapshot_download

# pnnbao-ump/VieNeu-TTS
# neuphonic/neucodec
# neuphonic/distill-neucodec
# neuphonic/neucodec-onnx-decoder-int8
# ntu-spml/distilhubert
# "facebook/w2v-bert-2.0"

models = [
    {"repo_id": "pnnbao-ump/VieNeu-TTS", "path_name": "models/VieNeu-TTS"},
    {"repo_id": "neuphonic/neucodec", "path_name": "src/models/tts-codec"},
    {"repo_id": "neuphonic/distill-neucodec", "path_name": "src/models/distill-tts-codec"},
    {"repo_id": "neuphonic/neucodec-onnx-decoder-int8", "path_name": "src/models/tts-codec-onnx-decoder-int8"},
    {"repo_id": "ntu-spml/distilhubert", "path_name": "src/models/distilhubert"},
    {"repo_id": "facebook/w2v-bert-2.0", "path_name": "src/models/w2v-bert-2.0"}
]

for model in models:
    snapshot_download(
        repo_id=model["repo_id"],
        local_dir=model["path_name"],
        local_dir_use_symlinks=False
    )