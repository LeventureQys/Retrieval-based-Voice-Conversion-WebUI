"""
Feature extraction for RVC-LoRA

This module provides feature extraction including:
- HuBERT phone features
- F0 (pitch) features using RMVPE
- Mel spectrogram features
"""

import os
import sys
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Add RVC root to path for imports
RVC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if RVC_ROOT not in sys.path:
    sys.path.insert(0, RVC_ROOT)


class HuBERTExtractor:
    """HuBERT feature extractor for phone features.

    Args:
        model_path: Path to HuBERT model (hubert_base.pt)
        device: Device to run on
        is_half: Whether to use half precision
        version: RVC version ("v1" or "v2")
    """

    def __init__(
        self,
        model_path: str = "assets/hubert/hubert_base.pt",
        device: str = "cuda",
        is_half: bool = True,
        version: str = "v2",
    ):
        self.device = device
        self.is_half = is_half
        self.version = version
        self.output_layer = 9 if version == "v1" else 12
        self.output_dim = 256 if version == "v1" else 768

        # Resolve model path
        if not os.path.isabs(model_path):
            # Try LoraModel/download first
            lora_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "download",
                os.path.basename(model_path)
            )
            if os.path.exists(lora_model_path):
                model_path = lora_model_path
            else:
                model_path = os.path.join(RVC_ROOT, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"HuBERT model not found: {model_path}\n"
                "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load HuBERT model."""
        try:
            import fairseq
        except ImportError:
            raise ImportError("fairseq is required for HuBERT. Install with: pip install fairseq")

        logger.info(f"Loading HuBERT model from {model_path}")

        # For PyTorch 2.6+, we need to allow unsafe loading for fairseq models
        # This is safe as we trust the HuBERT model from official sources
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [model_path],
                suffix="",
            )
        finally:
            torch.load = original_load

        self.model = models[0]
        self.model = self.model.to(self.device)
        self.saved_cfg = saved_cfg

        if self.is_half and self.device not in ["mps", "cpu"]:
            self.model = self.model.half()

        self.model.eval()
        logger.info(f"HuBERT model loaded (output_dim={self.output_dim})")

    @torch.no_grad()
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract HuBERT features from audio.

        Args:
            audio: Audio waveform at 16kHz sample rate

        Returns:
            Phone features [T, D] where D is 256 (v1) or 768 (v2)
        """
        # Convert to tensor
        feats = torch.from_numpy(audio).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)

        # Apply layer norm if required
        if self.saved_cfg.task.normalize:
            feats = F.layer_norm(feats, feats.shape)

        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)

        # Move to device
        if self.is_half and self.device not in ["mps", "cpu"]:
            feats = feats.half()
        feats = feats.to(self.device)
        padding_mask = padding_mask.to(self.device)

        # Extract features
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": self.output_layer,
        }

        logits = self.model.extract_features(**inputs)

        # Apply final projection for v1
        if self.version == "v1":
            feats = self.model.final_proj(logits[0])
        else:
            feats = logits[0]

        feats = feats.squeeze(0).float().cpu().numpy()

        if np.isnan(feats).sum() > 0:
            logger.warning("NaN values in HuBERT features")

        return feats


class F0Extractor:
    """F0 (pitch) extractor using RMVPE.

    Args:
        model_path: Path to RMVPE model
        device: Device to run on
        is_half: Whether to use half precision
        sample_rate: Audio sample rate (default: 16000)
        hop_size: Hop size in samples (default: 160)
    """

    def __init__(
        self,
        model_path: str = "assets/rmvpe/rmvpe.pt",
        device: str = "cuda",
        is_half: bool = True,
        sample_rate: int = 16000,
        hop_size: int = 160,
    ):
        self.device = device
        self.is_half = is_half
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        # F0 parameters
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Resolve model path
        if not os.path.isabs(model_path):
            model_path = os.path.join(RVC_ROOT, model_path)

        self.model_path = model_path
        self.rmvpe = None

    def _load_model(self):
        """Lazy load RMVPE model."""
        if self.rmvpe is not None:
            return

        try:
            from infer.lib.rmvpe import RMVPE
        except ImportError:
            raise ImportError(
                "RMVPE module not found. Make sure you're running from RVC root directory."
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"RMVPE model not found: {self.model_path}\n"
                "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )

        logger.info(f"Loading RMVPE model from {self.model_path}")
        self.rmvpe = RMVPE(self.model_path, is_half=self.is_half, device=self.device)
        logger.info("RMVPE model loaded")

    def extract(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """Extract F0 from audio.

        Args:
            audio: Audio waveform at 16kHz
            threshold: Voicing threshold

        Returns:
            F0 values in Hz [T]
        """
        self._load_model()
        f0 = self.rmvpe.infer_from_audio(audio, thred=threshold)
        return f0

    def f0_to_coarse(self, f0: np.ndarray) -> np.ndarray:
        """Convert continuous F0 to coarse (quantized) F0.

        Args:
            f0: Continuous F0 values in Hz

        Returns:
            Quantized F0 values [1-255]
        """
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse

    def extract_with_coarse(
        self,
        audio: np.ndarray,
        threshold: float = 0.03,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both continuous and coarse F0.

        Args:
            audio: Audio waveform at 16kHz
            threshold: Voicing threshold

        Returns:
            Tuple of (coarse_f0, continuous_f0)
        """
        f0 = self.extract(audio, threshold)
        f0_coarse = self.f0_to_coarse(f0)
        return f0_coarse, f0


class SpecExtractor:
    """Linear spectrogram extractor (for RVC training).

    RVC uses linear spectrograms, not mel spectrograms.
    Output dimension is n_fft // 2 + 1 (e.g., 1025 for n_fft=2048).

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size (filter_length)
        hop_size: Hop size
        win_size: Window size
    """

    def __init__(
        self,
        sample_rate: int = 40000,
        n_fft: int = 2048,
        hop_size: int = 400,
        win_size: int = 2048,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.spec_channels = n_fft // 2 + 1  # 1025 for n_fft=2048

        # Cache for hann window
        self._hann_window = None

    def _get_hann_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create Hann window."""
        if self._hann_window is None:
            self._hann_window = torch.hann_window(self.win_size)
        return self._hann_window.to(dtype=dtype, device=device)

    @torch.no_grad()
    def extract(
        self,
        audio: np.ndarray,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Extract linear spectrogram from audio.

        Args:
            audio: Audio waveform at target sample rate
            device: Device to run on

        Returns:
            Linear spectrogram [n_fft//2+1, T] (e.g., [1025, T])
        """
        # Convert to tensor
        y = torch.from_numpy(audio).float().unsqueeze(0)
        y = y.to(device)

        # Padding
        y = F.pad(
            y.unsqueeze(1),
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        # STFT
        hann_window = self._get_hann_window(y.device, y.dtype)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Magnitude (linear spectrogram)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

        return spec.squeeze(0)


class MelExtractor:
    """Mel spectrogram extractor.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        num_mels: Number of mel bands
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
    """

    def __init__(
        self,
        sample_rate: int = 40000,
        n_fft: int = 2048,
        num_mels: int = 128,
        hop_size: int = 400,
        win_size: int = 2048,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate // 2

        # Cache for mel basis and hann window
        self._mel_basis = None
        self._hann_window = None

    def _get_mel_basis(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create mel filterbank."""
        if self._mel_basis is None:
            try:
                from librosa.filters import mel as librosa_mel_fn
            except ImportError:
                raise ImportError("librosa is required for mel spectrogram extraction")

            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.num_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            self._mel_basis = torch.from_numpy(mel)

        return self._mel_basis.to(dtype=dtype, device=device)

    def _get_hann_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create Hann window."""
        if self._hann_window is None:
            self._hann_window = torch.hann_window(self.win_size)
        return self._hann_window.to(dtype=dtype, device=device)

    @torch.no_grad()
    def extract(
        self,
        audio: np.ndarray,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Extract mel spectrogram from audio.

        Args:
            audio: Audio waveform at target sample rate
            device: Device to run on

        Returns:
            Mel spectrogram [num_mels, T]
        """
        # Convert to tensor
        y = torch.from_numpy(audio).float().unsqueeze(0)
        y = y.to(device)

        # Padding
        y = F.pad(
            y.unsqueeze(1),
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        # STFT
        hann_window = self._get_hann_window(y.device, y.dtype)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Magnitude
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

        # Mel filterbank
        mel_basis = self._get_mel_basis(spec.device, spec.dtype)
        melspec = torch.matmul(mel_basis, spec)

        # Log compression
        melspec = torch.log(torch.clamp(melspec, min=1e-5))

        return melspec.squeeze(0)


class FeatureExtractor:
    """Combined feature extractor for RVC-LoRA.

    Extracts all features needed for training:
    - HuBERT phone features
    - F0 (pitch) features
    - Mel spectrogram

    Args:
        hubert_path: Path to HuBERT model
        rmvpe_path: Path to RMVPE model
        device: Device to run on
        is_half: Whether to use half precision
        version: RVC version ("v1" or "v2")
        sample_rate: Target sample rate for mel extraction
    """

    def __init__(
        self,
        hubert_path: str = "assets/hubert/hubert_base.pt",
        rmvpe_path: str = "assets/rmvpe/rmvpe.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_half: bool = True,
        version: str = "v2",
        sample_rate: int = 40000,
    ):
        self.device = device
        self.is_half = is_half
        self.version = version
        self.sample_rate = sample_rate

        # Initialize extractors (lazy loading)
        self._hubert_path = hubert_path
        self._rmvpe_path = rmvpe_path
        self._hubert = None
        self._f0 = None
        self._mel = None
        self._spec = None

    @property
    def hubert(self) -> HuBERTExtractor:
        """Get HuBERT extractor (lazy load)."""
        if self._hubert is None:
            self._hubert = HuBERTExtractor(
                model_path=self._hubert_path,
                device=self.device,
                is_half=self.is_half,
                version=self.version,
            )
        return self._hubert

    @property
    def f0_extractor(self) -> F0Extractor:
        """Get F0 extractor (lazy load)."""
        if self._f0 is None:
            self._f0 = F0Extractor(
                model_path=self._rmvpe_path,
                device=self.device,
                is_half=self.is_half,
            )
        return self._f0

    @property
    def mel_extractor(self) -> MelExtractor:
        """Get mel extractor (lazy load)."""
        if self._mel is None:
            self._mel = MelExtractor(sample_rate=self.sample_rate)
        return self._mel

    @property
    def spec_extractor(self) -> SpecExtractor:
        """Get linear spectrogram extractor (lazy load)."""
        if self._spec is None:
            self._spec = SpecExtractor(sample_rate=self.sample_rate)
        return self._spec

    def extract_hubert(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract HuBERT features.

        Args:
            audio_16k: Audio at 16kHz

        Returns:
            Phone features [T, D]
        """
        return self.hubert.extract(audio_16k)

    def extract_f0(
        self,
        audio_16k: np.ndarray,
        threshold: float = 0.03,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 features.

        Args:
            audio_16k: Audio at 16kHz
            threshold: Voicing threshold

        Returns:
            Tuple of (coarse_f0, continuous_f0)
        """
        return self.f0_extractor.extract_with_coarse(audio_16k, threshold)

    def extract_mel(
        self,
        audio: np.ndarray,
    ) -> torch.Tensor:
        """Extract mel spectrogram.

        Args:
            audio: Audio at target sample rate

        Returns:
            Mel spectrogram [num_mels, T]
        """
        return self.mel_extractor.extract(audio, device=self.device)

    def extract_spec(
        self,
        audio: np.ndarray,
    ) -> torch.Tensor:
        """Extract linear spectrogram (for RVC training).

        Args:
            audio: Audio at target sample rate

        Returns:
            Linear spectrogram [n_fft//2+1, T] (e.g., [1025, T])
        """
        return self.spec_extractor.extract(audio, device=self.device)

    def extract_all(
        self,
        audio_gt: np.ndarray,
        audio_16k: np.ndarray,
        f0_threshold: float = 0.03,
    ) -> Dict[str, Any]:
        """Extract all features from audio.

        Args:
            audio_gt: Audio at target sample rate (e.g., 40kHz)
            audio_16k: Audio at 16kHz
            f0_threshold: Voicing threshold for F0

        Returns:
            Dictionary containing:
            - phone: HuBERT features [T, D]
            - pitch: Coarse F0 [T]
            - pitchf: Continuous F0 [T]
            - spec: Linear spectrogram [n_fft//2+1, T] (e.g., [1025, T])
        """
        # Extract features
        phone = self.extract_hubert(audio_16k)
        pitch, pitchf = self.extract_f0(audio_16k, f0_threshold)
        spec = self.extract_spec(audio_gt)  # Use linear spectrogram, not mel

        # Align lengths (HuBERT hop=320 at 16kHz, spec hop=400 at 40kHz)
        # HuBERT: T = len(audio_16k) / 320
        # Mel: T = len(audio_gt) / 400
        # For 40kHz audio, 16kHz version is 2.5x shorter
        # HuBERT frames: len(audio_16k) / 320 = len(audio_gt) / 2.5 / 320 = len(audio_gt) / 800
        # Mel frames: len(audio_gt) / 400
        # Ratio: mel_frames / hubert_frames = 800 / 400 = 2

        min_len = min(phone.shape[0], pitch.shape[0], spec.shape[-1])

        return {
            'phone': phone[:min_len],
            'pitch': pitch[:min_len],
            'pitchf': pitchf[:min_len],
            'spec': spec[..., :min_len],
        }
