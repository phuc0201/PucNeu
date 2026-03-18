from .cuda import TTSWithCuda
from .utils import _compile_codec_with_triton

class TTSWithCudaPerformance(TTSWithCuda):
    def __init__(
            self,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self._logger(f"Initializing TTS with CUDA performance mode")

       
    def _load_codec(self, enable_triton):
        from .codec import codec
        self.codec = codec(device="cuda", mode="performance")
        if enable_triton:
            self._triton_enabled = _compile_codec_with_triton(self.codec)

