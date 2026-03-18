from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Generator
import json
import numpy as np
from huggingface_hub import hf_hub_download
import torch

from .utils.normalize_text import VietnameseTTSNormalizer
from .logger import Logger


class BaseTTS(ABC):
    def __init__(self,
        use_logger: bool = False,
        logger: Optional[Dict[str, Any]] = None,
    ):
        self.logger = None
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480

        self._preset_voices: Dict[str, Any] = {}
        self._default_voice: Optional[str] = None
        self.normalizer = VietnameseTTSNormalizer()
        self._ref_phoneme_cache: Dict[str, str] = {}

        if use_logger:
            if logger is None:
                logger = Logger.get_default_config()
            self.logger = Logger(
                log_dir=logger.get("log_dir"),
                print_to_console=logger.get("print_to_console"),
            ).get_logger()

        self.watermarker = None
        self._init_watermarker()

    def _logger(self, message: str, level: str = "info") -> None:
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None)
        if callable(log_method):
            log_method(message)
        else:
            self.logger.info(message)

    def _init_watermarker(self) -> None:
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
            self._logger("Audio watermarking initialized (Perth)")
        except (ImportError, AttributeError):
            self.watermarker = None

    def _apply_watermark(self, wav: np.ndarray) -> np.ndarray:
        if self.watermarker:
            return self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        return wav
    
    def _load_voices(self, local_model_path):
        try:
            file_path = Path(local_model_path) / "voices.json"
            if not file_path.exists():
                self._logger(f"Voice file not found: {file_path}", level="warning")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    self._logger(f"Invalid JSON in voice file {file_path}: {e}", level="error")
                    return

            if "presets" in data:
                self._preset_voices.update(data["presets"])
                self._logger(f"Loaded {len(data['presets'])} voices from {file_path}")

            if "default_voice" in data and data["default_voice"]:
                self._default_voice = data["default_voice"]

        except Exception as e:
            self._logger(f"Failed to load voices from {file_path}: {e}", level="error")

    def _resolve_ref_voice(
        self,
        voice: Optional[Dict[str, Any]] = None,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_text: Optional[str] = None
    ) -> tuple[Union[np.ndarray, torch.Tensor], str]:
        """Resolve reference voice codes and text."""
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)

        if ref_audio is not None and ref_codes is None:
            ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            try:
                voice_data = self.get_preset_voice(None)
                ref_codes = voice_data['codes']
                ref_text = voice_data['text']
            except Exception:
                pass

        if ref_codes is None or ref_text is None:
            raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        return ref_codes, ref_text

    def encode_reference(self, ref_audio_path: Union[str, Path]) -> torch.Tensor:
        import librosa
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]

        # Ensure device and dtype compatibility
        if hasattr(self.codec, "device"):
            wav_tensor = wav_tensor.to(self.codec.device)

        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def get_preset_voice(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        if voice_name is None:
            voice_name = self._default_voice
            if voice_name is None:
                if self._preset_voices:
                    voice_name = next(iter(self._preset_voices))
                else:
                    raise ValueError("No voice specified and no preset voices available.")

        if voice_name not in self._preset_voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {self.list_preset_voices()}")

        voice_data = self._preset_voices[voice_name]
        codes = voice_data["codes"]
        if isinstance(codes, list):
            codes = torch.tensor(codes, dtype=torch.long)
        return {"codes": codes, "text": voice_data["text"]}

    def get_ref_phonemes(self, ref_text: str) -> str:
        if ref_text not in self._ref_phoneme_cache:
            from .utils.phonemize_text import phonemize_with_dict
            self._ref_phoneme_cache[ref_text] = phonemize_with_dict(ref_text)
        return self._ref_phoneme_cache[ref_text]
    
    def list_preset_voices(self) -> List[tuple[str, str]]:
        """List available preset voices as (description, id)."""
        return [
            (v.get("description", k) if isinstance(v, dict) else str(v), k)
            for k, v in self._preset_voices.items()
        ]

    def save(self, audio: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save audio waveform to a file."""
        import soundfile as sf
        sf.write(str(output_path), audio, self.sample_rate)

    @abstractmethod
    def infer(self, text: str, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def infer_batch(self, texts: List[str], apply_watermark: bool = True, **kwargs: Any) -> List[np.ndarray]:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> 'BaseTTS':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass