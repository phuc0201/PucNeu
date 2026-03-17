import logging
import re
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from neucodec import DistillNeuCodec
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("VietEngine")


class VietEngine:
    BACKBONE_REPO = "models/VieNeu-TTS"       
    CODEC_REPO    = "models/distill-neucodec"   
    MAX_CONTEXT   = 2048
    SAMPLE_RATE   = 24_000

    def __init__(
        self,
        backbone_repo: str = BACKBONE_REPO,
        codec_repo: str = CODEC_REPO,
        device: str = "cpu",
        mode: str = "standard",
        hf_token: str = None,
    ):
        if mode not in ("standard", "fast"):
            raise ValueError(f"mode must be 'standard' or 'fast', got {mode!r}")
        
        self.mode = mode
        self.device = device
        self._phoneme_cache: dict = {}
        self.fast_pipeline = None
        self.fast_gen_config = None

        logger.info(f"Loading tokenizer  : {backbone_repo}")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading NeuCodec   : {codec_repo}  [{device}]")
        self.codec = self._load_local_distill_codec(codec_repo)

        if mode == "standard":
            logger.info(f"Mode: standard")
            self._load_standard_backbone(backbone_repo)
        else:
            logger.info(f"Mode: fast")
            if not self._init_fast_mode(backbone_repo):
                logger.warning("Fast mode initialization failed, falling back to standard")
                self.mode = "standard"
                self._load_standard_backbone(backbone_repo)

        self._init_phonemizer()
        logger.info("VietEngine ready.")

    def _load_standard_backbone(self, backbone_repo: str) -> None:
        """Load transformers backbone for standard mode."""
        logger.info(f"Loading backbone   : {backbone_repo}  [{self.device}]")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_repo,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.backbone.eval()

    def _init_fast_mode(self, backbone_repo: str) -> bool:
        """Initialize LMDeploy pipeline for fast mode (CUDA only). Returns True on success."""
        if self.device != "cuda":
            logger.warning(f"Fast mode only available on CUDA, current device: {self.device}")
            return False
        
        try:
            from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
            logger.info("Initializing LMDeploy pipeline...")
            engine_cfg = TurbomindEngineConfig(dtype="bfloat16")
            self.fast_pipeline = pipeline(backbone_repo, backend_config=engine_cfg)
            self.fast_gen_config = GenerationConfig(
                top_p=0.95,
                top_k=50,
                temperature=1.0,
                max_new_tokens=2048,
                min_new_tokens=40,
                do_sample=True,
                stop_words=["<|SPEECH_GENERATION_END|>"],
                skip_special_tokens=False,
            )
            logger.info("LMDeploy pipeline initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize LMDeploy: {e}")
            return False

    def _load_local_distill_codec(self, codec_repo: Union[str, Path]) -> DistillNeuCodec:
        """Load DistillNeuCodec from local snapshot (pytorch_model.bin)."""
        codec_path = Path(codec_repo)
        ckpt_path = codec_path / "pytorch_model.bin"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Codec checkpoint not found: {ckpt_path}")

        codec = DistillNeuCodec(24_000, 480)
        state_dict = torch.load(str(ckpt_path), map_location=self.device)
        codec.load_state_dict(state_dict, strict=False)
        return codec.eval().to(self.device)

    # ── Phonemizer ─────────────────────────────────────────────────────────────

    def _init_phonemizer(self) -> None:
        from sea_g2p import Normalizer, SEAPipeline
        self._pipeline   = SEAPipeline(lang="vi")
        self._normalizer = Normalizer()

    def _phonemize(self, text: str) -> str:
        if text not in self._phoneme_cache:
            self._phoneme_cache[text] = self._pipeline.run(text)
        return self._phoneme_cache[text]

    def _normalize(self, text: str) -> str:
        return self._normalizer.normalize(text)

    # ── Encode reference audio ─────────────────────────────────────────────────

    def encode_reference(self, ref_audio_path: Union[str, Path]) -> torch.Tensor:
        """WAV → 16kHz mono → NeuCodec.encode_code() → 1D LongTensor"""
        wav, _ = librosa.load(str(ref_audio_path), sr=16_000, mono=True)
        wav_t  = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            codes = self.codec.encode_code(audio_or_path=wav_t).squeeze(0).squeeze(0)
        return codes  # [T]

    # ── Build prompt ────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        ref_codes:    torch.Tensor,
        ref_phonemes: str,
        tgt_phonemes: str,
    ) -> list[int]:
        """
        Chat template:
          user: Convert the text to speech:
                <|TEXT_PROMPT_START|>{ref_phonemes} {tgt_phonemes}<|TEXT_PROMPT_END|>
          assistant:<|SPEECH_GENERATION_START|><|speech_0|><|speech_1|>...
        """
        tok = self.tokenizer

        SPEECH_REPLACE = tok.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        SPEECH_GEN_START = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        TEXT_REPLACE = tok.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        TEXT_PROMPT_START = tok.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        TEXT_PROMPT_END = tok.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        phoneme_ids = tok.encode(
            f"{ref_phonemes} {tgt_phonemes}",
            add_special_tokens=False,
        )

        ids = tok.encode(
            "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"
        )

        tr_idx = ids.index(TEXT_REPLACE)
        ids = (
            ids[:tr_idx]
            + [TEXT_PROMPT_START]
            + phoneme_ids
            + [TEXT_PROMPT_END]
            + ids[tr_idx + 1:]
        )

        sr_idx = ids.index(SPEECH_REPLACE)
        codes_tokens = tok.encode(
            "".join(f"<|speech_{i}|>" for i in ref_codes.flatten().tolist()),
            add_special_tokens=False,
        )
        ids = ids[:sr_idx] + [SPEECH_GEN_START] + codes_tokens
        logger.info("Build prompt: ", ids)

        return ids

    def _build_prompt_text(
        self,
        ref_codes:    torch.Tensor,
        ref_phonemes: str,
        tgt_phonemes: str,
    ) -> str:
        """Build text-based prompt for fast mode with special tokens."""
        speech_tokens_str = "".join(f"<|speech_{i}|>" for i in ref_codes.flatten().tolist())
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>"
            f"{ref_phonemes} {tgt_phonemes}<|TEXT_PROMPT_END|>\n"
            f"assistant:<|SPEECH_GENERATION_START|>{speech_tokens_str}"
        )
        return prompt

    # ── Decode output ───────────────────────────────────────────────────────────

    def _decode_output(self, output_str: str) -> np.ndarray:
        """speech token string → NeuCodec.decode_code() → float32 waveform"""
        ids = [int(m) for m in re.findall(r"<\|speech_(\d+)\|>", output_str)]
        if not ids:
            raise ValueError("No speech tokens found in model output.")

        codes = torch.tensor(ids, dtype=torch.long)[None, None, :].to(self.device)
        with torch.no_grad():
            recon = self.codec.decode_code(codes)
            if hasattr(recon, "cpu"):    recon = recon.cpu()
            if hasattr(recon, "numpy"): recon = recon.numpy()
        return recon[0, 0, :]

    # ── Inference methods ──────────────────────────────────────────────────────

    def _infer_standard(
        self,
        ref_codes:    torch.Tensor,
        ref_phonemes: str,
        tgt_phonemes: str,
        temperature:  float = 1.0,
        top_k:        int = 50,
    ) -> str:
        """Standard mode inference using transformers backbone."""
        prompt_ids = self._build_prompt(ref_codes, ref_phonemes, tgt_phonemes)
        prompt_t   = torch.tensor(prompt_ids).unsqueeze(0).to(self.device)

        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            out = self.backbone.generate(
                prompt_t,
                max_length=self.MAX_CONTEXT,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                repetition_penalty=1.05,
                use_cache=True,
                min_new_tokens=50,
            )
        
        gen_str = self.tokenizer.decode(
            out[0, prompt_t.shape[-1]:].cpu().tolist(),
            add_special_tokens=False,
        )
        return gen_str

    def _infer_fast(
        self,
        ref_codes:    torch.Tensor,
        ref_phonemes: str,
        tgt_phonemes: str,
        temperature:  float = 1.0,
        top_k:        int = 50,
    ) -> str:
        """Fast mode inference using LMDeploy pipeline."""
        if self.fast_pipeline is None or self.fast_gen_config is None:
            raise RuntimeError("LMDeploy pipeline not initialized for fast mode")
        
        prompt = self._build_prompt_text(ref_codes, ref_phonemes, tgt_phonemes)
        self.fast_gen_config.temperature = temperature
        self.fast_gen_config.top_k = top_k
        
        response = self.fast_pipeline([prompt], gen_config=self.fast_gen_config, do_preprocess=False)
        
        if isinstance(response, list):
            if not response:
                raise RuntimeError("LMDeploy returned empty response")
            first_response = response[0]
        else:
            first_response = response
        
        if hasattr(first_response, "text"):
            return first_response.text
        
        raise RuntimeError("Unexpected LMDeploy response format")

    # ── Public API ──────────────────────────────────────────────────────────────

    def clone(
        self,
        ref_audio_path: Union[str, Path],
        ref_text:       str,
        target_text:    str,
        temperature:    float = 1.0,
        top_k:          int   = 50,
    ) -> np.ndarray:
        """
        Zero-shot voice cloning.
        Returns float32 waveform array (sample_rate = SAMPLE_RATE = 24000).
        """
        ref_codes    = self.encode_reference(ref_audio_path)
        ref_phonemes = self._phonemize(self._normalize(ref_text))
        tgt_phonemes = self._phonemize(self._normalize(target_text))

        if self.mode == "fast" and self.fast_pipeline is not None:
            try:
                gen_str = self._infer_fast(ref_codes, ref_phonemes, tgt_phonemes, temperature, top_k)
            except Exception as e:
                logger.warning(f"Fast mode inference failed ({e}), falling back to standard")
                gen_str = self._infer_standard(ref_codes, ref_phonemes, tgt_phonemes, temperature, top_k)
        else:
            gen_str = self._infer_standard(ref_codes, ref_phonemes, tgt_phonemes, temperature, top_k)

        wav = self._decode_output(gen_str)
        return wav / (np.max(np.abs(wav)) + 1e-8)   # normalise to [-1, 1]

    def save(self, wav: np.ndarray, path: Union[str, Path]) -> None:
        """Save float32 waveform to WAV file."""
        sf.write(str(path), wav, self.SAMPLE_RATE)
        logger.info(f"Saved → {path}")