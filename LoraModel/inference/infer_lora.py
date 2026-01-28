"""
LoRA Inference Script for RVC

This script provides inference functionality for LoRA-enhanced RVC models.
"""

import os
import sys
import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.model_loader import load_model_for_inference, get_model_info

logger = logging.getLogger(__name__)


class LoRAInference:
    """Inference class for LoRA-enhanced RVC models.

    Args:
        model_path: Path to base model
        lora_path: Path to LoRA weights (optional)
        device: Device to run inference on
        is_half: Whether to use half precision
        version: Model version ("v1" or "v2")
        f0: Whether model supports F0
    """

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_half: bool = False,
        version: str = "v2",
        f0: bool = True,
    ):
        self.device = device
        self.is_half = is_half
        self.f0 = f0

        # Load model
        logger.info(f"Loading model from {model_path}")
        if lora_path:
            logger.info(f"Loading LoRA weights from {lora_path}")

        self.model, self.model_info = load_model_for_inference(
            model_path=model_path,
            lora_path=lora_path,
            device=device,
            is_half=is_half,
            version=version,
            f0=f0,
            merge_lora=True,
        )

        # Get the actual synthesizer
        if hasattr(self.model, 'synthesizer'):
            self.synthesizer = self.model.synthesizer
        else:
            self.synthesizer = self.model

        self.synthesizer.eval()
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def infer(
        self,
        phone: np.ndarray,
        phone_lengths: np.ndarray,
        pitch: np.ndarray,
        pitchf: np.ndarray,
        speaker_id: int = 0,
    ) -> np.ndarray:
        """Run inference on extracted features.

        Args:
            phone: Phone features [T, D]
            phone_lengths: Phone sequence length
            pitch: Quantized pitch [T]
            pitchf: Continuous pitch [T]
            speaker_id: Speaker ID

        Returns:
            Generated audio waveform
        """
        # Convert to tensors
        phone = torch.from_numpy(phone).unsqueeze(0).to(self.device)
        phone_lengths = torch.tensor([phone_lengths], dtype=torch.long).to(self.device)
        pitch = torch.from_numpy(pitch).unsqueeze(0).long().to(self.device)
        pitchf = torch.from_numpy(pitchf).unsqueeze(0).float().to(self.device)
        sid = torch.tensor([speaker_id], dtype=torch.long).to(self.device)

        if self.is_half:
            phone = phone.half()
            pitchf = pitchf.half()

        # Run inference
        if self.f0:
            audio, _, _ = self.synthesizer.infer(
                phone, phone_lengths, pitch, pitchf, sid
            )
        else:
            audio, _, _ = self.synthesizer.infer(
                phone, phone_lengths, sid
            )

        # Convert to numpy
        audio = audio.squeeze().cpu().numpy()

        return audio

    @torch.no_grad()
    def infer_from_audio(
        self,
        source_audio: np.ndarray,
        sample_rate: int,
        speaker_id: int = 0,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_path: Optional[str] = None,
        index_rate: float = 0.0,
        protect: float = 0.33,
    ) -> Tuple[np.ndarray, int]:
        """Run full inference pipeline from source audio.

        Args:
            source_audio: Source audio waveform (numpy array)
            sample_rate: Sample rate of source audio
            speaker_id: Target speaker ID
            f0_up_key: Pitch shift in semitones
            f0_method: F0 extraction method (currently only "rmvpe" supported)
            index_path: Path to index file (optional, not yet implemented)
            index_rate: Index retrieval rate (not yet implemented)
            protect: Protect voiceless consonants (0-0.5)

        Returns:
            Tuple of (generated_audio, output_sample_rate)
        """
        import torch.nn.functional as F
        from scipy import signal

        # High-pass filter coefficients for 16kHz
        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            try:
                import librosa
                audio_16k = librosa.resample(source_audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                from scipy.signal import resample
                num_samples = int(len(source_audio) * 16000 / sample_rate)
                audio_16k = resample(source_audio, num_samples).astype(np.float32)
        else:
            audio_16k = source_audio.astype(np.float32)

        # Apply high-pass filter
        audio_16k = signal.filtfilt(bh, ah, audio_16k)

        # Load HuBERT model if not loaded
        if not hasattr(self, '_hubert_model'):
            self._load_hubert()

        # Extract HuBERT features
        feats = self._extract_hubert(audio_16k)

        # Store for protection
        if protect < 0.5 and self.f0:
            feats0 = feats.clone()

        # Interpolate features (2x upsampling)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and self.f0:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # Extract F0 if needed
        if self.f0:
            pitch, pitchf = self._extract_f0(audio_16k, f0_up_key)

            # Align lengths
            p_len = min(feats.shape[1], pitch.shape[1])
            feats = feats[:, :p_len, :]
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

            # Apply protection
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
        if self.f0:
            audio_out = self.synthesizer.infer(feats, p_len_tensor, pitch, pitchf, sid)[0][0, 0]
        else:
            audio_out = self.synthesizer.infer(feats, p_len_tensor, sid)[0][0, 0]

        audio_out = audio_out.data.cpu().float().numpy()

        # Output sample rate (typically 40000 for RVC)
        output_sr = 40000

        return audio_out, output_sr

    def _load_hubert(self):
        """Load HuBERT model for feature extraction."""
        try:
            import fairseq
        except ImportError:
            raise ImportError("fairseq is required. Install with: pip install fairseq")

        # Find HuBERT model
        hubert_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "download", "hubert_base.pt"),
            "assets/hubert/hubert_base.pt",
        ]

        hubert_path = None
        for p in hubert_paths:
            if os.path.exists(p):
                hubert_path = p
                break

        if hubert_path is None:
            raise FileNotFoundError("HuBERT model not found")

        logger.info(f"Loading HuBERT from {hubert_path}")
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path], suffix=""
        )
        self._hubert_model = models[0].to(self.device)
        self._hubert_cfg = saved_cfg

        if self.is_half and self.device not in ["mps", "cpu"]:
            self._hubert_model = self._hubert_model.half()
        self._hubert_model.eval()

    def _extract_hubert(self, audio_16k: np.ndarray) -> torch.Tensor:
        """Extract HuBERT features."""
        import torch.nn.functional as F

        feats = torch.from_numpy(audio_16k).float()
        if self._hubert_cfg.task.normalize:
            feats = F.layer_norm(feats, feats.shape)

        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)

        if self.is_half and self.device not in ["mps", "cpu"]:
            feats = feats.half()
        feats = feats.to(self.device)
        padding_mask = padding_mask.to(self.device)

        version = "v2"  # Default to v2
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }

        with torch.no_grad():
            logits = self._hubert_model.extract_features(**inputs)
            if version == "v1":
                feats = self._hubert_model.final_proj(logits[0])
            else:
                feats = logits[0]

        return feats

    def _extract_f0(self, audio_16k: np.ndarray, f0_up_key: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract F0 features."""
        # Load RMVPE if not loaded
        if not hasattr(self, '_rmvpe'):
            rmvpe_paths = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "download", "rmvpe.pt"),
                "assets/rmvpe/rmvpe.pt",
            ]
            rmvpe_path = None
            for p in rmvpe_paths:
                if os.path.exists(p):
                    rmvpe_path = p
                    break

            if rmvpe_path is None:
                raise FileNotFoundError("RMVPE model not found")

            from infer.lib.rmvpe import RMVPE
            self._rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device)

        # Extract F0
        f0 = self._rmvpe.infer_from_audio(audio_16k, thred=0.03)

        # Apply pitch shift
        f0 *= pow(2, f0_up_key / 12)

        # Convert to coarse F0
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        pitch = torch.from_numpy(f0_coarse).unsqueeze(0).long().to(self.device)
        pitchf = torch.from_numpy(f0).unsqueeze(0).float().to(self.device)

        if self.is_half and self.device not in ["mps", "cpu"]:
            pitchf = pitchf.half()

        return pitch, pitchf


def extract_features(
    audio_path: str,
    f0_method: str = "rmvpe",
    hop_length: int = 160,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from audio file.

    This is a placeholder that would integrate with RVC's feature extraction.

    Args:
        audio_path: Path to audio file
        f0_method: F0 extraction method
        hop_length: Hop length for feature extraction

    Returns:
        Tuple of (phone, phone_lengths, pitch, pitchf)
    """
    try:
        from infer.modules.vc.utils import extract_features as rvc_extract
        return rvc_extract(audio_path, f0_method, hop_length)
    except ImportError:
        raise ImportError(
            "RVC feature extraction not available. "
            "Please provide pre-extracted features."
        )


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description='LoRA Inference for RVC')

    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to base model checkpoint')
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA weights (optional)')

    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input features (.npz) or audio file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output audio file')

    # Model parameters
    parser.add_argument('--version', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='RVC model version')
    parser.add_argument('--f0', action='store_true', default=True,
                        help='Use F0 model')
    parser.add_argument('--no_f0', action='store_false', dest='f0',
                        help='Use non-F0 model')

    # Inference parameters
    parser.add_argument('--speaker_id', type=int, default=0,
                        help='Speaker ID')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision')

    # Info mode
    parser.add_argument('--info', action='store_true',
                        help='Print model info and exit')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Info mode
    if args.info:
        info = get_model_info(args.model)
        print("\nModel Information:")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key}: {value}")
        if args.lora:
            print("\nLoRA Information:")
            print("-" * 40)
            lora_info = get_model_info(args.lora)
            for key, value in lora_info.items():
                print(f"  {key}: {value}")
        return 0

    # Create inference instance
    inference = LoRAInference(
        model_path=args.model,
        lora_path=args.lora,
        device=args.device,
        is_half=args.half,
        version=args.version,
        f0=args.f0,
    )

    # Load input features
    if args.input.endswith('.npz'):
        # Load pre-extracted features
        data = np.load(args.input)
        phone = data['phone']
        phone_lengths = data['phone_lengths'] if 'phone_lengths' in data else len(phone)
        pitch = data['pitch'] if 'pitch' in data else np.zeros(len(phone), dtype=np.int64)
        pitchf = data['pitchf'] if 'pitchf' in data else np.zeros(len(phone), dtype=np.float32)
    else:
        # Try to extract features from audio
        logger.info(f"Extracting features from {args.input}")
        phone, phone_lengths, pitch, pitchf = extract_features(args.input)

    # Run inference
    logger.info("Running inference...")
    audio = inference.infer(
        phone=phone,
        phone_lengths=phone_lengths,
        pitch=pitch,
        pitchf=pitchf,
        speaker_id=args.speaker_id,
    )

    # Save output
    logger.info(f"Saving output to {args.output}")

    # Determine sample rate from model config
    sample_rate = 40000  # Default RVC sample rate

    try:
        import soundfile as sf
        sf.write(args.output, audio, sample_rate)
    except ImportError:
        try:
            from scipy.io import wavfile
            # Normalize to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(args.output, sample_rate, audio_int16)
        except ImportError:
            # Save as numpy
            np.save(args.output.replace('.wav', '.npy'), audio)
            logger.warning(f"Saved as numpy array (soundfile/scipy not available)")

    logger.info("Inference completed!")
    return 0


if __name__ == '__main__':
    exit(main())
