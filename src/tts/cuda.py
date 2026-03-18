import os
from abc import abstractmethod
from typing import Optional, Union, List, Dict, Any
import torch
import numpy as np
from pathlib import Path
import gc

from huggingface_hub import snapshot_download

from .utils import extract_speech_ids
from .base_tts import BaseTTS
from .utils.core_utils import split_text_into_chunks, join_audio_chunks
from .utils.phonemize_text import phonemize_batch

MODELS_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "../models/")

class TTSWithCuda(BaseTTS):
    def __init__(
            self,
            repo: Optional[str] = None,
            hf_token: Optional[str] = None,
            local_model_path: Optional[str] = None,
            local_files_only: bool = False,
            tp: int = 1,
            memory_util: float = 0.3,
            enable_prefix_caching: bool = True,
            quant_policy: int = 0,
            enable_triton: bool = True,
            max_batch_size: int = 4,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.local_model_path = local_model_path
        self.codec = None        

        self._load_model(repo, local_model_path=self.local_model_path, local_files_only=local_files_only, memory_util=memory_util, tp=tp, enable_prefix_caching=enable_prefix_caching, quant_policy=quant_policy, enable_triton=enable_triton, max_batch_size=max_batch_size, hf_token=hf_token)
        self._load_voices(self.local_model_path)

    def _load_model(self, repo, local_model_path, local_files_only, memory_util, tp, enable_prefix_caching, quant_policy, enable_triton, max_batch_size, hf_token):
        try:
            from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        except ImportError as e:
            raise ImportError("Failed to import LMDeploy. Please make sure you have LMDeploy installed and properly set up. ") from e
    
        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=memory_util,
            tp=tp,
            enable_prefix_caching=enable_prefix_caching,
            dtype='bfloat16',
            quant_policy=quant_policy,
        )

        self.model = pipeline(local_model_path, backend_config=backend_config, local_files_only=local_files_only, token=hf_token)

        self.gen_config = GenerationConfig(
            top_p=0.95, top_k=50, temperature=1.0, max_new_tokens=2048,
            do_sample=True, min_new_tokens=40,
        )

    def _warmup_model(self):
        self._logger("Warming up the model with dummy input...")
        try:
            dummy_codes = list(range(10))
            dummy_prompt = self._format_prompt(dummy_codes, "warmup", "test")
            _ = self.model([dummy_prompt], gen_config=self.gen_config, do_preprocess=False)
            self._logger("Model warmup completed successfully!")
        except Exception as e:
            self._logger(f"Model warmup failed: {e}. This may indicate an issue with the model or LMDeploy setup.", level="warning")

    def _format_prompt(
        self,
        ref_codes: Union[List[int], torch.Tensor, np.ndarray],
        ref_text: str,
        input_text: str,
        ref_phonemes: Optional[str] = None,
        input_phonemes: Optional[str] = None
    ) -> str:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes_list = ref_codes.flatten().tolist()
        else:
            ref_codes_list = ref_codes

        from .utils.phonemize_text import phonemize_with_dict

        ref_text_phones = ref_phonemes if ref_phonemes else self.get_ref_phonemes(ref_text)
        input_text_phones = input_phonemes if input_phonemes else phonemize_with_dict(input_text, skip_normalize=True)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])

        return (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

    
    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> np.ndarray:
        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            prompt = self._format_prompt(ref_codes, ref_text, chunks[0])
            responses = self.model([prompt], gen_config=self.gen_config, do_preprocess=False)
            wav = self._decode(responses[0].text)
            wav = self._apply_watermark(wav)
        else:
            all_wavs = self.infer_batch(chunks, ref_codes, ref_text, voice=voice, temperature=temperature, top_k=top_k, skip_normalize=True)
            wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
        return wav
    
    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, max_batch_size: Optional[int] = None) -> List[np.ndarray]:
        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        max_batch_size = max_batch_size or self.max_batch_size

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        # Pre-phonemize all for performance
        ref_phonemes = self.get_ref_phonemes(ref_text)
        chunk_phonemes = phonemize_batch(texts, skip_normalize=True)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            batch_phonemes = chunk_phonemes[i : i + max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, text, ref_phonemes=ref_phonemes, input_phonemes=ph)
                      for text, ph in zip(batch_texts, batch_phonemes)]
            responses = self.model(prompts, gen_config=self.gen_config, do_preprocess=False)
            batch_codes = [response.text for response in responses]
            batch_wavs = [self._decode(codes) for codes in batch_codes]
            if apply_watermark:
                batch_wavs = [self._apply_watermark(w) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
        return all_wavs

    def _decode(self, codes_str: str) -> np.ndarray:
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            raise ValueError("No valid speech tokens found in the output. This may indicate an issue with the model output or decoding process.")
        with torch.no_grad():
            codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
            recon = self.codec.decode_code(codes).cpu().numpy()
        return recon[0, 0, :]
    
    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @abstractmethod
    def _load_codec(self, enable_triton: bool):
        pass