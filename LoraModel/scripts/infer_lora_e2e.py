#!/usr/bin/env python
"""
End-to-End LoRA Inference Script for RVC

This script provides complete voice conversion using LoRA-enhanced RVC models:
1. Load source audio
2. Extract features (HuBERT, F0)
3. Apply voice conversion with LoRA
4. Save converted audio

Usage:
    python infer_lora_e2e.py --source input.wav --output output.wav --model base.pth --lora lora.pth
"""

import os
import sys
import argparse
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.io import wavfile

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_ROOT = os.path.dirname(SCRIPT_DIR)
RVC_ROOT = os.path.dirname(LORA_ROOT)

sys.path.insert(0, LORA_ROOT)
sys.path.insert(0, RVC_ROOT)

logger = logging.getLogger(__name__)

# High-pass filter for 16kHz audio
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


class LoRAVoiceConverter:
    """End-to-end voice converter using LoRA-enhanced RVC models.

    Args:
        base_model_path: Path to base RVC model
        lora_path: Path to LoRA weights (optional)
        hubert_path: Path to HuBERT model
        rmvpe_path: Path to RMVPE model
        device: Device to run on
        is_half: Whether to use half precision
        version: RVC version ("v1" or "v2")
        f0: Whether model uses F0
        sample_rate: Target sample rate
    """

    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[str] = None,
        hubert_path: str = "assets/hubert/hubert_base.pt",
        rmvpe_path: str = "assets/rmvpe/rmvpe.pt",
        device: str = "cuda",
        is_half: bool = True,
        version: str = "v2",
        f0: bool = True,
        sample_rate: int = 40000,
    ):
        self.device = device
        self.is_half = is_half
        self.version = version
        self.use_f0 = f0
        self.sample_rate = sample_rate

        # HuBERT parameters
        self.hubert_sr = 16000
        self.hubert_hop = 320  # HuBERT hop size at 16kHz

        # F0 parameters
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Load models
        self._load_hubert(hubert_path)
        if f0:
            self._load_rmvpe(rmvpe_path)
        self._load_synthesizer(base_model_path, lora_path)

        logger.info("LoRA Voice Converter initialized")

    def _find_model_path(self, path: str, candidates: list) -> str:
        """Find model path from candidates."""
        if os.path.exists(path):
            return path
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Model not found: {path}")

    def _load_hubert(self, hubert_path: str):
        """Load HuBERT model."""
        candidates = [
            hubert_path,
            os.path.join(LORA_ROOT, "download", "hubert_base.pt"),
            os.path.join(RVC_ROOT, "assets", "hubert", "hubert_base.pt"),
        ]
        hubert_path = self._find_model_path(hubert_path, candidates)

        logger.info(f"Loading HuBERT from {hubert_path}")

        try:
            import fairseq
        except ImportError:
            raise ImportError("fairseq is required. Install with: pip install fairseq")

        # For PyTorch 2.6+, we need to allow unsafe loading for fairseq models
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [hubert_path], suffix=""
            )
        finally:
            torch.load = original_load

        self.hubert = models[0].to(self.device)
        self.hubert_cfg = saved_cfg

        if self.is_half and self.device not in ["mps", "cpu"]:
            self.hubert = self.hubert.half()

        self.hubert.eval()

    def _load_rmvpe(self, rmvpe_path: str):
        """Load RMVPE model for F0 extraction."""
        candidates = [
            rmvpe_path,
            os.path.join(LORA_ROOT, "download", "rmvpe.pt"),
            os.path.join(RVC_ROOT, "assets", "rmvpe", "rmvpe.pt"),
        ]
        rmvpe_path = self._find_model_path(rmvpe_path, candidates)

        logger.info(f"Loading RMVPE from {rmvpe_path}")

        from infer.lib.rmvpe import RMVPE
        self.rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device)

    def _load_synthesizer(self, base_model_path: str, lora_path: Optional[str]):
        """Load synthesizer with optional LoRA."""
        logger.info(f"Loading synthesizer from {base_model_path}")

        from models import load_synthesizer_with_lora
        from lora import LoRAConfig

        # Create LoRA config if loading LoRA
        lora_config = None
        if lora_path and os.path.exists(lora_path):
            # Try to load config from checkpoint
            checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
            if 'lora_config' in checkpoint:
                cfg = checkpoint['lora_config']
                if isinstance(cfg, dict):
                    lora_config = LoRAConfig(**cfg)
                else:
                    lora_config = cfg
            elif 'config' in checkpoint:
                cfg = checkpoint['config']
                if isinstance(cfg, dict):
                    lora_config = LoRAConfig(**cfg)
                else:
                    lora_config = cfg
            else:
                lora_config = LoRAConfig(r=8, lora_alpha=16)

        # Load model
        self.model = load_synthesizer_with_lora(
            checkpoint_path=base_model_path,
            lora_config=lora_config,
            device=self.device,
            version=self.version,
            f0=self.use_f0,
        )

        # Load LoRA weights if provided
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LoRA weights from {lora_path}")
            from lora import load_lora_weights
            # Load checkpoint file first
            lora_checkpoint = torch.load(lora_path, map_location=self.device, weights_only=False)
            # Extract lora_weights from checkpoint
            if isinstance(lora_checkpoint, dict) and 'lora_weights' in lora_checkpoint:
                lora_weights = lora_checkpoint['lora_weights']
            else:
                lora_weights = lora_checkpoint
            load_lora_weights(self.model.synthesizer, lora_weights)

        # Get synthesizer
        self.synthesizer = self.model.synthesizer
        self.synthesizer.eval()

        if self.is_half and self.device not in ["mps", "cpu"]:
            self.synthesizer = self.synthesizer.half()

    def load_audio(self, audio_path: str, target_sr: int) -> np.ndarray:
        """Load audio file at target sample rate."""
        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio.astype(np.float32)
        except ImportError:
            # Fallback to scipy + resampling
            sr, audio = wavfile.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                from scipy.signal import resample
                num_samples = int(len(audio) * target_sr / sr)
                audio = resample(audio, num_samples).astype(np.float32)
            return audio

    @torch.no_grad()
    def extract_hubert_features(self, audio_16k: np.ndarray) -> torch.Tensor:
        """Extract HuBERT features from 16kHz audio."""
        # Convert to tensor (ensure contiguous array)
        feats = torch.from_numpy(audio_16k.copy()).float()

        # Apply layer norm if required
        if self.hubert_cfg.task.normalize:
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
            "output_layer": 9 if self.version == "v1" else 12,
        }

        logits = self.hubert.extract_features(**inputs)

        if self.version == "v1":
            feats = self.hubert.final_proj(logits[0])
        else:
            feats = logits[0]

        return feats

    @torch.no_grad()
    def extract_f0(
        self,
        audio_16k: np.ndarray,
        f0_up_key: int = 0,
        threshold: float = 0.03,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract F0 from 16kHz audio.

        Args:
            audio_16k: Audio at 16kHz
            f0_up_key: Pitch shift in semitones
            threshold: Voicing threshold

        Returns:
            Tuple of (pitch, pitchf) tensors
        """
        # Extract F0 using RMVPE (ensure contiguous array)
        f0 = self.rmvpe.infer_from_audio(audio_16k.copy(), thred=threshold)

        # Apply pitch shift
        f0 *= pow(2, f0_up_key / 12)

        # Convert to coarse F0
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        # Convert to tensors
        pitch = torch.from_numpy(f0_coarse).unsqueeze(0).long().to(self.device)
        pitchf = torch.from_numpy(f0).unsqueeze(0).float().to(self.device)

        if self.is_half and self.device not in ["mps", "cpu"]:
            pitchf = pitchf.half()

        return pitch, pitchf

    @torch.no_grad()
    def convert(
        self,
        source_audio_path: str,
        f0_up_key: int = 0,
        speaker_id: int = 0,
        protect: float = 0.33,
    ) -> Tuple[np.ndarray, int]:
        """Convert voice from source audio.

        Args:
            source_audio_path: Path to source audio file
            f0_up_key: Pitch shift in semitones
            speaker_id: Target speaker ID
            protect: Protect voiceless consonants (0-0.5)

        Returns:
            Tuple of (converted_audio, sample_rate)
        """
        logger.info(f"Loading source audio: {source_audio_path}")

        # Load audio at 16kHz for feature extraction
        audio_16k = self.load_audio(source_audio_path, 16000)

        # Apply high-pass filter
        audio_16k = signal.filtfilt(bh, ah, audio_16k)

        logger.info("Extracting HuBERT features...")
        feats = self.extract_hubert_features(audio_16k)

        # Store original features for protection
        if protect < 0.5 and self.use_f0:
            feats0 = feats.clone()

        # Interpolate features (2x upsampling to match pitch)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if protect < 0.5 and self.use_f0:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # Extract F0 if needed
        if self.use_f0:
            logger.info("Extracting F0...")
            pitch, pitchf = self.extract_f0(audio_16k, f0_up_key)

            # Align lengths
            p_len = min(feats.shape[1], pitch.shape[1])
            feats = feats[:, :p_len, :]
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

            # Apply protection for voiceless consonants
            if protect < 0.5:
                feats0 = feats0[:, :p_len, :]
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)
        else:
            pitch = None
            pitchf = None
            p_len = feats.shape[1]

        # Prepare inputs
        p_len_tensor = torch.tensor([p_len], device=self.device).long()
        sid = torch.tensor([speaker_id], device=self.device).long()

        # Run inference
        logger.info("Running voice conversion...")
        if self.use_f0:
            audio_out = self.synthesizer.infer(feats, p_len_tensor, pitch, pitchf, sid)[0][0, 0]
        else:
            audio_out = self.synthesizer.infer(feats, p_len_tensor, sid)[0][0, 0]

        # Convert to numpy
        audio_out = audio_out.data.cpu().float().numpy()

        return audio_out, self.sample_rate

    def convert_file(
        self,
        source_path: str,
        output_path: str,
        f0_up_key: int = 0,
        speaker_id: int = 0,
        protect: float = 0.33,
    ) -> str:
        """Convert audio file and save result.

        Args:
            source_path: Path to source audio
            output_path: Path for output audio
            f0_up_key: Pitch shift in semitones
            speaker_id: Target speaker ID
            protect: Protect voiceless consonants

        Returns:
            Path to output file
        """
        # Convert
        audio, sr = self.convert(source_path, f0_up_key, speaker_id, protect)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.99

        # Save
        logger.info(f"Saving output to {output_path}")

        try:
            import soundfile as sf
            sf.write(output_path, audio, sr)
        except ImportError:
            # Convert to int16 and save with scipy
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sr, audio_int16)

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End LoRA Voice Conversion for RVC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python infer_lora_e2e.py -s input.wav -o output.wav -m base.pth -l lora.pth

  # With pitch shift
  python infer_lora_e2e.py -s input.wav -o output.wav -m base.pth -l lora.pth --pitch 2

  # Without LoRA (base model only)
  python infer_lora_e2e.py -s input.wav -o output.wav -m base.pth
        """
    )

    # Required arguments
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Path to source audio file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path for output audio file')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to base RVC model')

    # Optional arguments
    parser.add_argument('--lora', '-l', type=str, default=None,
                        help='Path to LoRA weights')
    parser.add_argument('--pitch', '-p', type=int, default=0,
                        help='Pitch shift in semitones (default: 0)')
    parser.add_argument('--speaker_id', type=int, default=0,
                        help='Speaker ID (default: 0)')
    parser.add_argument('--protect', type=float, default=0.33,
                        help='Protect voiceless consonants 0-0.5 (default: 0.33)')

    # Model options
    parser.add_argument('--version', '-v', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='RVC version (default: v2)')
    parser.add_argument('--f0', action='store_true', default=True,
                        help='Use F0 model (default: True)')
    parser.add_argument('--no_f0', action='store_false', dest='f0',
                        help='Use non-F0 model')
    parser.add_argument('--sample_rate', '-sr', type=int, default=40000,
                        choices=[32000, 40000, 48000],
                        help='Sample rate (default: 40000)')

    # Device options
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--no_half', action='store_true',
                        help='Disable half precision')

    # Feature model paths
    parser.add_argument('--hubert', type=str, default='assets/hubert/hubert_base.pt',
                        help='Path to HuBERT model')
    parser.add_argument('--rmvpe', type=str, default='assets/rmvpe/rmvpe.pt',
                        help='Path to RMVPE model')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        args.no_half = True

    # Check source file
    if not os.path.exists(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        return 1

    # Check model file
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return 1

    try:
        # Create converter
        converter = LoRAVoiceConverter(
            base_model_path=args.model,
            lora_path=args.lora,
            hubert_path=args.hubert,
            rmvpe_path=args.rmvpe,
            device=args.device,
            is_half=not args.no_half,
            version=args.version,
            f0=args.f0,
            sample_rate=args.sample_rate,
        )

        # Convert
        output_path = converter.convert_file(
            source_path=args.source,
            output_path=args.output,
            f0_up_key=args.pitch,
            speaker_id=args.speaker_id,
            protect=args.protect,
        )

        print(f"\nConversion complete!")
        print(f"Output saved to: {output_path}")
        return 0

    except Exception as e:
        logger.exception(f"Conversion failed: {e}")
        print(f"\nERROR: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
