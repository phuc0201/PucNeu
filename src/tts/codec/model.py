from abc import ABC, abstractmethod
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class GPUCodec(nn.Module, PyTorchModelHubMixin, ABC):
    def __init__(
        self,
        sample_rate: int = 24_000,
        hop_length: int = 480,
    ):
        super().__init__()
        self.model_folder_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../models/",
        )
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        from .codec_decoder_vocos import CodecDecoderVocos

        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_post_a = nn.Linear(2048, 1024)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _extract_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        extracted = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not extracted:
            raise KeyError(f"Missing weights for prefix: {prefix}")
        return extracted

    def _load_decoder_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.generator.load_state_dict(
            self._extract_state_dict(state_dict, "generator."),
            strict=True,
        )
        self.fc_post_a.load_state_dict(
            self._extract_state_dict(state_dict, "fc_post_a."),
            strict=True,
        )

    def _prepare_audio(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        if isinstance(audio_or_path, (Path, str)):
            import torchaudio
            from torchaudio import transforms as T

            y, sr = torchaudio.load(audio_or_path)

            if y.dim() == 2 and y.shape[0] > 1:
                y = y[:1, :]

            if sr != 16_000:
                y = T.Resample(sr, 16_000)(y)

            y = y.unsqueeze(0)

        elif isinstance(audio_or_path, torch.Tensor):
            y = audio_or_path
            if y.dim() != 3:
                raise ValueError(
                    f"Codec expects tensor audio input to be of shape [B, 1, T], got: {tuple(y.shape)}"
                )
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_or_path)}")

        pad_for_wav = (320 - (y.shape[-1] % 320)) % 320
        if pad_for_wav > 0:
            y = torch.nn.functional.pad(y, (0, pad_for_wav))

        return y.float()

    @abstractmethod
    def encode_code(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def decode_code(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        fsq_post_emb = self.generator.quantizer.get_output_from_indices(
            fsq_codes.transpose(1, 2)
        )
        fsq_post_emb = fsq_post_emb.transpose(1, 2)
        fsq_post_emb = self.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon = self.generator(fsq_post_emb.transpose(1, 2), vq=False)[0]
        return recon


class GPUPerformanceCodec(GPUCodec):
    def __init__(
        self,
        sample_rate: int = 24_000,
        hop_length: int = 480,
        device: str = "cuda",
    ):
        super().__init__(sample_rate, hop_length)

        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        from .codec_encoder import CodecEncoder
        from .module import SemanticEncoder

        encode_model_path = os.path.join(self.model_folder_path, "w2v-bert-2.0")
        ckpt_path = os.path.join(self.model_folder_path, "tts-codec", "pytorch_model.bin")

        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            encode_model_path,
            output_hidden_states=True,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(encode_model_path)
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.CodecEnc = CodecEncoder()
        self.fc_prior = nn.Linear(2048, 2048)

        state_dict = torch.load(ckpt_path, map_location="cpu")

        self.CodecEnc.load_state_dict(
            self._extract_state_dict(state_dict, "CodecEnc."),
            strict=True,
        )
        self.SemanticEncoder_module.load_state_dict(
            self._extract_state_dict(state_dict, "SemanticEncoder_module."),
            strict=True,
        )
        self.fc_prior.load_state_dict(
            self._extract_state_dict(state_dict, "fc_prior."),
            strict=True,
        )
        self._load_decoder_weights(state_dict)

        self.to(device)
        self.eval()

    @torch.inference_mode()
    def encode_code(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        y = self._prepare_audio(audio_or_path)

        semantic_features = self.feature_extractor(
            y.squeeze(0),
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features.to(self.device)

        acoustic_emb = self.CodecEnc(y.to(self.device))
        acoustic_emb = acoustic_emb.transpose(1, 2)

        semantic_output = self.semantic_model(semantic_features).hidden_states[16]
        semantic_output = semantic_output.transpose(1, 2)
        semantic_encoded = self.SemanticEncoder_module(semantic_output)

        min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
        acoustic_emb = acoustic_emb[:, :, :min_len]
        semantic_encoded = semantic_encoded[:, :, :min_len]

        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes


class GPUBalanceCodec(GPUCodec):
    def __init__(
        self,
        sample_rate: int = 24_000,
        hop_length: int = 480,
        device: str = "cuda",
    ):
        super().__init__(sample_rate, hop_length)

        from transformers import AutoFeatureExtractor, HubertModel
        from .codec_encoder_distill import DistillCodecEncoder
        from .module import SemanticEncoder

        encode_model_path = os.path.join(self.model_folder_path, "distilhubert")
        ckpt_path = os.path.join(self.model_folder_path, "distill-tts-codec", "pytorch_model.bin")

        self.semantic_model = HubertModel.from_pretrained(
            encode_model_path,
            output_hidden_states=True,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(encode_model_path)
        self.SemanticEncoder_module = SemanticEncoder(768, 768, 1024)
        self.codec_encoder = DistillCodecEncoder()
        self.fc_prior = nn.Linear(768 + 768, 2048)
        self.fc_sq_prior = nn.Linear(512, 768)

        state_dict = torch.load(ckpt_path, map_location="cpu")

        self.codec_encoder.load_state_dict(
            self._extract_state_dict(state_dict, "codec_encoder."),
            strict=True,
        )
        self.SemanticEncoder_module.load_state_dict(
            self._extract_state_dict(state_dict, "SemanticEncoder_module."),
            strict=True,
        )
        self.fc_prior.load_state_dict(
            self._extract_state_dict(state_dict, "fc_prior."),
            strict=True,
        )
        self.fc_sq_prior.load_state_dict(
            self._extract_state_dict(state_dict, "fc_sq_prior."),
            strict=True,
        )
        self._load_decoder_weights(state_dict)

        self.to(device)
        self.eval()

    @torch.inference_mode()
    def encode_code(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        y = self._prepare_audio(audio_or_path)

        semantic_features = (
            self.feature_extractor(
                torch.nn.functional.pad(y[0, :].cpu(), (160, 160)),
                sampling_rate=16_000,
                return_tensors="pt",
            )
            .input_values.to(self.device)
            .squeeze(0)
        )

        fsq_emb = self.fc_sq_prior(self.codec_encoder(y.to(self.device)))
        fsq_emb = fsq_emb.transpose(1, 2)

        semantic_target = self.semantic_model(semantic_features).last_hidden_state
        semantic_target = semantic_target.transpose(1, 2)
        semantic_target = self.SemanticEncoder_module(semantic_target)

        min_len = min(fsq_emb.shape[-1], semantic_target.shape[-1])
        fsq_emb = fsq_emb[:, :, :min_len]
        semantic_target = semantic_target[:, :, :min_len]

        concat_emb = torch.cat([semantic_target, fsq_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes


class CPUCodec:
    def __init__(self, onnx_path: str | None = None):
        self.onnx_path = onnx_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../models/tts-codec-onnx-decoder-int8/model.onnx",
        )

        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed. Please install it with: pip install onnxruntime"
            ) from exc

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            self.onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def decode_code(self, codes: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().numpy()

        if not isinstance(codes, np.ndarray):
            raise ValueError("Codes should be a numpy array or torch tensor.")
        if codes.ndim != 3 or codes.shape[1] != 1:
            raise ValueError(f"Codes should have shape [B, 1, F], got: {codes.shape}")

        if codes.dtype != np.int32:
            codes = codes.astype(np.int32)

        recon = self.session.run(None, {"codes": codes})[0].astype(np.float32)
        return recon