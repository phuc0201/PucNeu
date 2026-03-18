
import os

from huggingface_hub import snapshot_download

def TTS(
        device: str = "cuda",
        mode: str = "balance",
        *args,
        **kwargs
    ):

    local_model_path = kwargs.get("local_model_path")
    local_files_only = kwargs.get("local_files_only")
    repo = kwargs.get("repo")
    hf_token = kwargs.get("hf_token")

    if local_files_only is not None:
        local_model_path = (
            local_model_path
            if os.path.isabs(local_model_path)
            else os.path.abspath(local_model_path)
        )
    elif repo is not None:
        local_model_path = os.path.join(os.path.dirname(__file__), "../", repo)
    else:
        raise ValueError("Either 'local_files_only' or 'repo' must be provided in kwargs.")
    
    kwargs["local_model_path"] = local_model_path

    if local_files_only is False and not os.path.exists(local_model_path):
        snapshot_download(repo, local_dir=local_model_path, local_dir_use_symlinks=False, token=hf_token)
    
    match device:
        case "cuda":
            match mode:
                case "balance":
                    from .cuda_balance import TTSWithCudaBalance
                    return TTSWithCudaBalance(*args, **kwargs)
                case "performance":
                    from .cuda_performance import TTSWithCudaPerformance
                    return TTSWithCudaPerformance(*args, **kwargs)
                case _:
                    raise ValueError(f"Unsupported mode '{mode}' for device 'cuda'. Supported modes: 'balance', 'performance'.")
        case "cpu":
            from .cpu import TTSWithCPU
            return TTSWithCPU(*args, **kwargs)
        case _:
            raise ValueError(f"Unsupported device '{device}'. Supported devices: 'cuda', 'cpu'.")