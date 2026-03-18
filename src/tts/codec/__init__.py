def codec(device, mode:str = "balance", *args, **kwargs):
    match device:
        case "cuda":
            match mode:
                case "balance":
                    from .model import GPUBalanceCodec
                    return GPUBalanceCodec(*args, **kwargs)
                case "performance":
                    from .model import GPUPerformanceCodec
                    return GPUPerformanceCodec(*args, **kwargs)
                case _:
                    raise ValueError(f"Unsupported mode: {mode}")
        case "cpu":
            from .model import CPUCodec
            return CPUCodec(*args, **kwargs)
        case _:
            raise ValueError(f"Unsupported device: {device}")