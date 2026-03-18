import torch
from typing import Optional, Union, List, Dict, Any, Generator
import platform
import os
from huggingface_hub import snapshot_download
import gc
from pathlib import Path
import numpy as np


from .base_tts import BaseTTS
from .utils.core_utils import split_text_into_chunks, join_audio_chunks
from .utils.phonemize_text import phonemize_with_dict, phonemize_batch


MODELS_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "../models/")

class TTSWithCPU(BaseTTS):
    def __init__(self,
        repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_files_only: bool = False,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.model = None
        self.local_model_path = local_model_path
        
        self._load_model(repo, self.local_model_path, local_files_only, hf_token) 
        self._load_codec()
        self._load_voices(self.local_model_path)
        self._warmup_model()


    def _load_model(self, repo, local_model_path, local_files_only, hf_token):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=local_files_only, token=hf_token)
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=local_files_only, token=hf_token).to(torch.device("cpu"))

        try:
            if platform.system() == "Linux":
                self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception as e:
            self._logger(f"Failed to compile the model with torch.compile: {e}. Proceeding without compilation.", level="warning")

    def _load_codec(self):
        from .codec import codec
        self.codec = codec(device="cpu")
        self._is_onnx_codec = True

    def _warmup_model(self):
        try:
            self._logger("Warming up standard model...")
            dummy_text = "Xin chào"
            # Using very short dummy ref to speed up
            dummy_ref_codes = torch.zeros(10, dtype=torch.long)
            dummy_ref_text = "Chào"
            _ = self.infer(dummy_text, ref_codes=dummy_ref_codes, ref_text=dummy_ref_text, max_chars=16)
            self._logger("Warmup complete")
        except Exception as e:
            self._logger(f"Warmup failed: {e}", level="warning")

    def _apply_chat_template(self, ref_codes: Union[List[int], torch.Tensor, np.ndarray], ref_phonemes: str, chunk_phonemes: str) -> List[int]:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes_list = ref_codes.flatten().tolist()
        else:
            ref_codes_list = ref_codes

        full_phonemes = f"{ref_phonemes} {chunk_phonemes}"

        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(full_phonemes, add_special_tokens=False)
        chat = "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = ids[:text_replace_idx] + [text_prompt_start] + input_ids + [text_prompt_end] + ids[text_replace_idx + 1:]

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes_list])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        return ids

    def _infer_torch(self, prompt_ids: List[int], temperature: float = 1.0, top_k: int = 50) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.model.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.model.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False)
        return output_str

    def _decode(self, codes_str: str) -> np.ndarray:
        from .utils import extract_speech_ids
        speech_ids = extract_speech_ids(codes_str)

        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in the output.")

        codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
        recon = self.codec.decode_code(codes)
        return recon[0, 0, :]

    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            ref_phonemes = self.get_ref_phonemes(ref_text)
            phonemes = phonemize_with_dict(chunks[0], skip_normalize=True)
            prompt_ids = self._apply_chat_template(ref_codes, ref_phonemes, phonemes)
            output_str = self._infer_torch(prompt_ids, temperature, top_k)
            wav = self._decode(output_str)
            return self._apply_watermark(wav)

        all_wavs = self.infer_batch(
            chunks,
            ref_codes=ref_codes,
            ref_text=ref_text,
            temperature=temperature,
            top_k=top_k,
            skip_normalize=True,
            apply_watermark=False
        )
        final_wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
        return self._apply_watermark(final_wav)
    
    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True) -> List[np.ndarray]:
        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        ref_phonemes = self.get_ref_phonemes(ref_text)
        chunk_phonemes = phonemize_batch(texts, skip_normalize=True)

        all_wavs = []

        batch_prompt_ids = []
        for phonemes in chunk_phonemes:
            prompt_ids = self._apply_chat_template(ref_codes, ref_phonemes, phonemes)
            batch_prompt_ids.append(torch.tensor(prompt_ids))

        inputs = self.tokenizer.pad(
            {"input_ids": batch_prompt_ids},
            padding=True,
            return_tensors="pt"
        )
        # Move all tensors to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                min_new_tokens=50,
            )

        input_length = inputs["input_ids"].shape[-1]
        for i in range(len(texts)):
            generated_ids = output_tokens[i, input_length:]
            output_str = self.tokenizer.decode(generated_ids, add_special_tokens=False)
            wav = self._decode(output_str)
            if apply_watermark:
                wav = self._apply_watermark(wav)
            all_wavs.append(wav)

        return all_wavs

    def close(self) -> None:
        """Explicitly release model resources."""
        try:
            if self.model is not None:
                self.model = None

            if self.codec is not None:
                self.codec = None

            torch_module = globals().get("torch")
            if torch_module is not None:
                cuda = getattr(torch_module, "cuda", None)
                if cuda is not None and cuda.is_available():
                    cuda.empty_cache()
        except Exception as e:
            self._logger(f"Error during StandardTTS closure: {e}", level="error")



