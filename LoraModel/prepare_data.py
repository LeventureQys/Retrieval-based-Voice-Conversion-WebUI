"""
RVC-LoRA Data Preparation Script

This script prepares training data for LoRA fine-tuning by extracting:
1. HuBERT features (phone features)
2. F0/pitch features (using RMVPE)
3. Mel spectrograms

Usage:
    python prepare_data.py --input_dir ./raw_audio --output_dir ./processed_data

The script will process all .wav files in input_dir and save extracted features to output_dir.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

# Add parent directory to path for RVC imports
SCRIPT_DIR = Path(__file__).resolve().parent
RVC_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RVC_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from audio for LoRA training."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_half: bool = True,
        version: str = "v2",  # v1 or v2
        f0_method: str = "rmvpe",
    ):
        self.device = device
        self.is_half = is_half and device == "cuda"
        self.version = version
        self.f0_method = f0_method

        self.sr = 16000  # HuBERT input sample rate
        self.hop_length = 320  # 160 for 100Hz, 320 for 50Hz
        self.window = 160  # F0 window

        # Models (lazy loading)
        self._hubert_model = None
        self._rmvpe_model = None

        logger.info(f"FeatureExtractor initialized: device={device}, version={version}, f0_method={f0_method}")

    @property
    def hubert_model(self):
        """Lazy load HuBERT model."""
        if self._hubert_model is None:
            self._hubert_model = self._load_hubert()
        return self._hubert_model

    @property
    def rmvpe_model(self):
        """Lazy load RMVPE model."""
        if self._rmvpe_model is None:
            self._rmvpe_model = self._load_rmvpe()
        return self._rmvpe_model

    def _load_hubert(self):
        """Load HuBERT model."""
        logger.info("Loading HuBERT model...")

        # Try different paths for HuBERT model
        hubert_paths = [
            RVC_ROOT / "assets" / "hubert" / "hubert_base.pt",
            SCRIPT_DIR / "download" / "hubert_base.pt",
            Path(os.environ.get("hubert_path", "")) if os.environ.get("hubert_path") else None,
        ]

        hubert_path = None
        for path in hubert_paths:
            if path and path.exists():
                hubert_path = path
                break

        if hubert_path is None:
            raise FileNotFoundError(
                "HuBERT model not found. Please download hubert_base.pt and place it in:\n"
                f"  - {RVC_ROOT / 'assets' / 'hubert' / 'hubert_base.pt'}\n"
                f"  - {SCRIPT_DIR / 'download' / 'hubert_base.pt'}"
            )

        logger.info(f"Loading HuBERT from: {hubert_path}")

        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [str(hubert_path)],
            suffix="",
        )
        hubert = models[0]
        hubert = hubert.to(self.device)

        if self.is_half:
            hubert = hubert.half()
        else:
            hubert = hubert.float()

        hubert.eval()
        return hubert

    def _load_rmvpe(self):
        """Load RMVPE model for F0 extraction."""
        logger.info("Loading RMVPE model...")

        # Try different paths
        rmvpe_paths = [
            RVC_ROOT / "assets" / "rmvpe" / "rmvpe.pt",
            SCRIPT_DIR / "download" / "rmvpe.pt",
            Path(os.environ.get("rmvpe_root", "")) / "rmvpe.pt" if os.environ.get("rmvpe_root") else None,
        ]

        rmvpe_path = None
        for path in rmvpe_paths:
            if path and path.exists():
                rmvpe_path = path
                break

        if rmvpe_path is None:
            raise FileNotFoundError(
                "RMVPE model not found. Please download rmvpe.pt and place it in:\n"
                f"  - {RVC_ROOT / 'assets' / 'rmvpe' / 'rmvpe.pt'}\n"
                f"  - {SCRIPT_DIR / 'download' / 'rmvpe.pt'}"
            )

        logger.info(f"Loading RMVPE from: {rmvpe_path}")

        from infer.lib.rmvpe import RMVPE
        rmvpe = RMVPE(str(rmvpe_path), is_half=self.is_half, device=self.device)
        return rmvpe

    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load and resample audio file."""
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)

    def extract_hubert_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract HuBERT features from audio.

        Args:
            audio: Audio waveform at 16kHz

        Returns:
            HuBERT features with shape (time_frames, feature_dim)
            - v1: (T, 256)
            - v2: (T, 768)
        """
        feats = torch.from_numpy(audio)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()

        feats = feats.view(1, -1).to(self.device)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if self.version == "v1" else 12,
        }

        with torch.no_grad():
            logits = self.hubert_model.extract_features(**inputs)
            if self.version == "v1":
                feats = self.hubert_model.final_proj(logits[0])
            else:
                feats = logits[0]

        feats = feats.squeeze(0).cpu().numpy()

        if self.is_half:
            feats = feats.astype(np.float32)

        return feats  # (T, 256) or (T, 768)

    def extract_f0(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 (pitch) from audio.

        Args:
            audio: Audio waveform at 16kHz

        Returns:
            Tuple of (f0_coarse, f0_continuous):
            - f0_coarse: Quantized pitch (1-255)
            - f0_continuous: Continuous F0 in Hz
        """
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if self.f0_method == "rmvpe":
            f0 = self.rmvpe_model.infer_from_audio(audio, thred=0.03)
        else:
            # Fallback to pyworld harvest
            import pyworld
            audio_double = audio.astype(np.float64)
            f0, t = pyworld.harvest(
                audio_double,
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,  # 10ms = 100Hz
            )
            f0 = pyworld.stonemask(audio_double, f0, t, self.sr)

        # Store continuous F0
        f0_continuous = f0.copy()

        # Convert to mel scale and quantize
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        return f0_coarse, f0_continuous

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 1024,
        n_mels: int = 128,
        hop_length: int = 320,
        win_length: int = 1024,
        fmin: float = 0,
        fmax: float = 8000,
    ) -> np.ndarray:
        """Extract mel spectrogram from audio.

        Args:
            audio: Audio waveform at 16kHz

        Returns:
            Mel spectrogram with shape (n_mels, time_frames)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # Log compression
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

        return mel.astype(np.float32)

    def process_audio(
        self,
        audio_path: str,
        output_dir: str,
        save_wav: bool = True,
    ) -> dict:
        """Process a single audio file and extract all features.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save extracted features
            save_wav: Whether to save the processed waveform

        Returns:
            Dictionary with paths to saved features
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get base name
        base_name = Path(audio_path).stem

        # Load audio
        logger.info(f"Processing: {audio_path}")
        audio = self.load_audio(audio_path, target_sr=self.sr)

        # Extract features
        logger.info("  Extracting HuBERT features...")
        phone_features = self.extract_hubert_features(audio)

        logger.info("  Extracting F0...")
        f0_coarse, f0_continuous = self.extract_f0(audio)

        logger.info("  Extracting mel spectrogram...")
        mel_spec = self.extract_mel_spectrogram(audio)

        # Align lengths (use phone features as reference, 50Hz)
        phone_len = phone_features.shape[0]

        # F0 should match phone length
        if len(f0_coarse) > phone_len:
            f0_coarse = f0_coarse[:phone_len]
            f0_continuous = f0_continuous[:phone_len]
        elif len(f0_coarse) < phone_len:
            f0_coarse = np.pad(f0_coarse, (0, phone_len - len(f0_coarse)), mode='edge')
            f0_continuous = np.pad(f0_continuous, (0, phone_len - len(f0_continuous)), mode='edge')

        # Mel spectrogram: adjust to match (mel is at 50Hz with hop_length=320)
        mel_len = mel_spec.shape[1]
        if mel_len > phone_len:
            mel_spec = mel_spec[:, :phone_len]
        elif mel_len < phone_len:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, phone_len - mel_len)), mode='edge')

        # Save features
        output_paths = {}

        # Phone features
        phone_path = os.path.join(output_dir, f"{base_name}_phone.npy")
        np.save(phone_path, phone_features)
        output_paths['phone'] = phone_path

        # F0 features
        pitch_path = os.path.join(output_dir, f"{base_name}_pitch.npy")
        np.save(pitch_path, f0_coarse)
        output_paths['pitch'] = pitch_path

        pitchf_path = os.path.join(output_dir, f"{base_name}_pitchf.npy")
        np.save(pitchf_path, f0_continuous)
        output_paths['pitchf'] = pitchf_path

        # Mel spectrogram
        spec_path = os.path.join(output_dir, f"{base_name}.spec.pt")
        torch.save(torch.from_numpy(mel_spec), spec_path)
        output_paths['spec'] = spec_path

        # Waveform
        if save_wav:
            wav_path = os.path.join(output_dir, f"{base_name}.wav.pt")
            torch.save(torch.from_numpy(audio), wav_path)
            output_paths['wav'] = wav_path

        logger.info(f"  Saved features: phone={phone_features.shape}, f0={f0_coarse.shape}, mel={mel_spec.shape}")

        return output_paths


def process_directory(
    input_dir: str,
    output_dir: str,
    device: str = "cuda",
    version: str = "v2",
    f0_method: str = "rmvpe",
    extensions: list = ['.wav', '.mp3', '.flac', '.ogg'],
):
    """Process all audio files in a directory.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save processed features
        device: Device to use (cuda/cpu)
        version: RVC version (v1/v2)
        f0_method: F0 extraction method
        extensions: Audio file extensions to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files")

    # Initialize extractor
    extractor = FeatureExtractor(
        device=device,
        version=version,
        f0_method=f0_method,
    )

    # Process each file
    successful = 0
    failed = 0

    for audio_path in audio_files:
        try:
            extractor.process_audio(str(audio_path), str(output_dir))
            successful += 1
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            traceback.print_exc()
            failed += 1

    # Create file list
    filelist_path = output_dir / "filelist.txt"
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for audio_path in audio_files:
            base_name = audio_path.stem
            f.write(f"{base_name}.wav\n")

    logger.info(f"\nProcessing complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  File list saved to: {filelist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for RVC-LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all audio files in a directory
    python prepare_data.py --input_dir ./raw_audio --output_dir ./processed_data

    # Use CPU instead of GPU
    python prepare_data.py --input_dir ./raw_audio --output_dir ./processed_data --device cpu

    # Use v1 model (256-dim features)
    python prepare_data.py --input_dir ./raw_audio --output_dir ./processed_data --version v1
        """
    )

    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing raw audio files"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory to save processed features"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="RVC version (v1: 256-dim, v2: 768-dim)"
    )
    parser.add_argument(
        "--f0_method",
        type=str,
        default="rmvpe",
        choices=["rmvpe", "harvest"],
        help="F0 extraction method"
    )

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        version=args.version,
        f0_method=args.f0_method,
    )


if __name__ == "__main__":
    main()
