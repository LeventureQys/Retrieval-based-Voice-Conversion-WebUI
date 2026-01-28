"""
End-to-end preprocessing pipeline for RVC-LoRA

This module provides a complete pipeline from raw audio to training-ready features.
"""

import os
import sys
import logging
import json
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from .audio_processor import AudioProcessor, load_audio
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for RVC-LoRA training.

    This pipeline handles:
    1. Audio loading and preprocessing (filtering, slicing, normalization)
    2. Feature extraction (HuBERT, F0, Mel spectrogram)
    3. Saving features in training-ready format

    Args:
        output_dir: Directory for saving processed data
        sample_rate: Target sample rate (default: 40000)
        version: RVC version ("v1" or "v2")
        device: Device for feature extraction
        is_half: Whether to use half precision
        hubert_path: Path to HuBERT model
        rmvpe_path: Path to RMVPE model
    """

    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 40000,
        version: str = "v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_half: bool = True,
        hubert_path: str = "assets/hubert/hubert_base.pt",
        rmvpe_path: str = "assets/rmvpe/rmvpe.pt",
    ):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.version = version
        self.device = device
        self.is_half = is_half

        # Create output directories
        self._setup_directories()

        # Initialize processors
        self.audio_processor = AudioProcessor(
            target_sr=sample_rate,
            slice_audio=True,
            segment_duration=3.7,
            overlap=0.3,
        )

        self.feature_extractor = FeatureExtractor(
            hubert_path=hubert_path,
            rmvpe_path=rmvpe_path,
            device=device,
            is_half=is_half,
            version=version,
            sample_rate=sample_rate,
        )

    def _setup_directories(self):
        """Create output directory structure."""
        self.gt_wavs_dir = os.path.join(self.output_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(self.output_dir, "1_16k_wavs")
        self.f0_dir = os.path.join(self.output_dir, "2a_f0")
        self.f0nsf_dir = os.path.join(self.output_dir, "2b_f0nsf")
        self.feature_dir = os.path.join(
            self.output_dir,
            "3_feature256" if self.version == "v1" else "3_feature768"
        )
        self.spec_dir = os.path.join(self.output_dir, "4_spec")
        self.training_dir = os.path.join(self.output_dir, "training_data")

        for d in [
            self.output_dir,
            self.gt_wavs_dir,
            self.wavs16k_dir,
            self.f0_dir,
            self.f0nsf_dir,
            self.feature_dir,
            self.spec_dir,
            self.training_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def process_audio_files(
        self,
        input_dir: str,
        extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a'),
    ) -> List[str]:
        """Process audio files: slice, normalize, and save.

        Args:
            input_dir: Directory containing audio files
            extensions: Valid audio file extensions

        Returns:
            List of base names for processed segments
        """
        logger.info(f"Processing audio files from {input_dir}")

        # Find all audio files
        audio_files = []
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(extensions):
                audio_files.append(os.path.join(input_dir, f))

        if not audio_files:
            raise ValueError(f"No audio files found in {input_dir}")

        logger.info(f"Found {len(audio_files)} audio files")

        processed_names = []
        segment_idx = 0

        for file_idx, file_path in enumerate(audio_files):
            try:
                # Load and filter audio
                audio = self.audio_processor.load_audio(file_path)
                audio = self.audio_processor.apply_highpass_filter(audio)

                # Slice audio
                if self.audio_processor.slice_audio:
                    chunks = self.audio_processor.slicer.slice(audio)
                else:
                    chunks = [audio]

                # Process each chunk
                for chunk in chunks:
                    segments = list(self.audio_processor._split_into_segments(chunk))

                    for segment in segments:
                        # Normalize
                        normalized = self.audio_processor.normalize(segment)
                        if normalized is None:
                            continue

                        # Generate base name
                        base_name = f"{file_idx}_{segment_idx}"

                        # Save at original sample rate
                        gt_path = os.path.join(self.gt_wavs_dir, f"{base_name}.wav")
                        wavfile.write(gt_path, self.sample_rate, normalized)

                        # Resample to 16kHz and save
                        audio_16k = self.audio_processor.resample(
                            normalized, self.sample_rate, 16000
                        )
                        wav16k_path = os.path.join(self.wavs16k_dir, f"{base_name}.wav")
                        wavfile.write(wav16k_path, 16000, audio_16k)

                        processed_names.append(base_name)
                        segment_idx += 1

                logger.info(f"Processed {file_path}")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        logger.info(f"Total segments: {len(processed_names)}")
        return processed_names

    def extract_features(
        self,
        base_names: Optional[List[str]] = None,
        f0_threshold: float = 0.03,
    ) -> List[str]:
        """Extract features for all processed audio segments.

        Args:
            base_names: List of base names to process (if None, process all)
            f0_threshold: Voicing threshold for F0 extraction

        Returns:
            List of successfully processed base names
        """
        # Get list of files to process
        if base_names is None:
            base_names = [
                f.replace('.wav', '')
                for f in os.listdir(self.wavs16k_dir)
                if f.endswith('.wav')
            ]

        logger.info(f"Extracting features for {len(base_names)} segments")

        successful = []

        for idx, base_name in enumerate(base_names):
            try:
                # Load audio files
                wav16k_path = os.path.join(self.wavs16k_dir, f"{base_name}.wav")
                gt_path = os.path.join(self.gt_wavs_dir, f"{base_name}.wav")

                # Read audio
                sr_16k, audio_16k = wavfile.read(wav16k_path)
                sr_gt, audio_gt = wavfile.read(gt_path)

                # Convert to float32 if needed
                if audio_16k.dtype == np.int16:
                    audio_16k = audio_16k.astype(np.float32) / 32768.0
                if audio_gt.dtype == np.int16:
                    audio_gt = audio_gt.astype(np.float32) / 32768.0

                # Extract HuBERT features
                phone = self.feature_extractor.extract_hubert(audio_16k)
                phone_path = os.path.join(self.feature_dir, f"{base_name}.npy")
                np.save(phone_path, phone, allow_pickle=False)

                # Extract F0
                pitch, pitchf = self.feature_extractor.extract_f0(audio_16k, f0_threshold)
                f0_path = os.path.join(self.f0_dir, f"{base_name}.npy")
                f0nsf_path = os.path.join(self.f0nsf_dir, f"{base_name}.npy")
                np.save(f0_path, pitch, allow_pickle=False)
                np.save(f0nsf_path, pitchf, allow_pickle=False)

                # Extract mel spectrogram
                spec = self.feature_extractor.extract_mel(audio_gt)
                spec_path = os.path.join(self.spec_dir, f"{base_name}.spec.pt")
                torch.save(spec.cpu(), spec_path)

                successful.append(base_name)

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(base_names)} segments")

            except Exception as e:
                logger.error(f"Failed to extract features for {base_name}: {e}")

        logger.info(f"Successfully extracted features for {len(successful)} segments")
        return successful

    def prepare_training_data(
        self,
        base_names: Optional[List[str]] = None,
    ) -> str:
        """Prepare training data in the format expected by LoRA trainer.

        Creates consolidated feature files in training_data directory.

        Args:
            base_names: List of base names to include (if None, use all)

        Returns:
            Path to filelist.txt
        """
        if base_names is None:
            base_names = [
                f.replace('.npy', '')
                for f in os.listdir(self.feature_dir)
                if f.endswith('.npy')
            ]

        logger.info(f"Preparing training data for {len(base_names)} segments")

        filelist = []

        for base_name in base_names:
            try:
                # Load all features
                phone_path = os.path.join(self.feature_dir, f"{base_name}.npy")
                f0_path = os.path.join(self.f0_dir, f"{base_name}.npy")
                f0nsf_path = os.path.join(self.f0nsf_dir, f"{base_name}.npy")
                spec_path = os.path.join(self.spec_dir, f"{base_name}.spec.pt")
                gt_path = os.path.join(self.gt_wavs_dir, f"{base_name}.wav")

                # Check all files exist
                if not all(os.path.exists(p) for p in [phone_path, f0_path, f0nsf_path, spec_path, gt_path]):
                    logger.warning(f"Missing files for {base_name}, skipping")
                    continue

                # Load features
                phone = np.load(phone_path)
                pitch = np.load(f0_path)
                pitchf = np.load(f0nsf_path)
                spec = torch.load(spec_path)

                # Load and convert wav
                sr, wav = wavfile.read(gt_path)
                if wav.dtype == np.int16:
                    wav = wav.astype(np.float32) / 32768.0
                wav = torch.from_numpy(wav)

                # Align lengths
                min_len = min(phone.shape[0], pitch.shape[0], spec.shape[-1])
                phone = phone[:min_len]
                pitch = pitch[:min_len]
                pitchf = pitchf[:min_len]
                spec = spec[..., :min_len]

                # Save consolidated training file
                output_path = os.path.join(self.training_dir, f"{base_name}.pt")
                torch.save({
                    'phone': torch.from_numpy(phone),
                    'pitch': torch.from_numpy(pitch).long(),
                    'pitchf': torch.from_numpy(pitchf).float(),
                    'spec': spec,
                    'wav': wav,
                }, output_path)

                filelist.append(base_name)

            except Exception as e:
                logger.error(f"Failed to prepare {base_name}: {e}")

        # Write filelist
        filelist_path = os.path.join(self.output_dir, "filelist.txt")
        with open(filelist_path, 'w') as f:
            for name in filelist:
                f.write(f"{name}\n")

        # Save metadata
        metadata = {
            'version': self.version,
            'sample_rate': self.sample_rate,
            'num_samples': len(filelist),
            'feature_dim': 256 if self.version == "v1" else 768,
        }
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Prepared {len(filelist)} training samples")
        logger.info(f"Filelist saved to {filelist_path}")

        return filelist_path

    def run(
        self,
        input_dir: str,
        f0_threshold: float = 0.03,
    ) -> Dict[str, Any]:
        """Run the complete preprocessing pipeline.

        Args:
            input_dir: Directory containing raw audio files
            f0_threshold: Voicing threshold for F0 extraction

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Version: {self.version}")
        logger.info(f"Sample rate: {self.sample_rate}")
        logger.info("=" * 50)

        # Step 1: Process audio files
        logger.info("\n[Step 1/3] Processing audio files...")
        base_names = self.process_audio_files(input_dir)

        if not base_names:
            raise RuntimeError("No audio segments were processed")

        # Step 2: Extract features
        logger.info("\n[Step 2/3] Extracting features...")
        successful = self.extract_features(base_names, f0_threshold)

        if not successful:
            raise RuntimeError("No features were extracted")

        # Step 3: Prepare training data
        logger.info("\n[Step 3/3] Preparing training data...")
        filelist_path = self.prepare_training_data(successful)

        logger.info("\n" + "=" * 50)
        logger.info("Preprocessing complete!")
        logger.info(f"Total samples: {len(successful)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Filelist: {filelist_path}")
        logger.info("=" * 50)

        return {
            'output_dir': self.output_dir,
            'filelist_path': filelist_path,
            'num_samples': len(successful),
            'base_names': successful,
        }


def preprocess_for_lora(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 40000,
    version: str = "v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    is_half: bool = True,
    hubert_path: str = "assets/hubert/hubert_base.pt",
    rmvpe_path: str = "assets/rmvpe/rmvpe.pt",
    f0_threshold: float = 0.03,
) -> Dict[str, Any]:
    """Convenience function to run preprocessing pipeline.

    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory for output
        sample_rate: Target sample rate
        version: RVC version
        device: Device for feature extraction
        is_half: Whether to use half precision
        hubert_path: Path to HuBERT model
        rmvpe_path: Path to RMVPE model
        f0_threshold: Voicing threshold for F0

    Returns:
        Dictionary with pipeline results
    """
    pipeline = PreprocessingPipeline(
        output_dir=output_dir,
        sample_rate=sample_rate,
        version=version,
        device=device,
        is_half=is_half,
        hubert_path=hubert_path,
        rmvpe_path=rmvpe_path,
    )

    return pipeline.run(input_dir, f0_threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio for RVC-LoRA training")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input directory containing audio files")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--sample_rate", "-sr", type=int, default=40000,
                        help="Target sample rate (default: 40000)")
    parser.add_argument("--version", "-v", type=str, default="v2",
                        choices=["v1", "v2"],
                        help="RVC version (default: v2)")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="Device for feature extraction (default: cuda)")
    parser.add_argument("--no_half", action="store_true",
                        help="Disable half precision")
    parser.add_argument("--hubert", type=str, default="assets/hubert/hubert_base.pt",
                        help="Path to HuBERT model")
    parser.add_argument("--rmvpe", type=str, default="assets/rmvpe/rmvpe.pt",
                        help="Path to RMVPE model")
    parser.add_argument("--f0_threshold", type=float, default=0.03,
                        help="F0 voicing threshold (default: 0.03)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run pipeline
    result = preprocess_for_lora(
        input_dir=args.input,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        version=args.version,
        device=args.device,
        is_half=not args.no_half,
        hubert_path=args.hubert,
        rmvpe_path=args.rmvpe,
        f0_threshold=args.f0_threshold,
    )

    print(f"\nPreprocessing complete!")
    print(f"Samples: {result['num_samples']}")
    print(f"Output: {result['output_dir']}")
