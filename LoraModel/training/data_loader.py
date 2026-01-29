"""
Data loading utilities for RVC-LoRA training

This module provides data loaders for training LoRA-enhanced RVC models.
"""

import os
import random
import logging
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class LoRATrainingDataset(Dataset):
    """Dataset for LoRA fine-tuning.

    Loads pre-extracted features for training:
    - Mel spectrograms (.spec.pt files)
    - Phone features (.npy files)
    - Pitch features (.npy files, optional)
    - Audio waveforms (.wav.pt files)

    Args:
        data_dir: Directory containing training data
        filelist_path: Path to file list (text file with audio paths)
        segment_size: Size of audio segments for training
        hop_length: Hop length used for feature extraction
        use_f0: Whether to use F0 (pitch) features
        speaker_id: Speaker ID for single-speaker training (default: 0)
    """

    def __init__(
        self,
        data_dir: str,
        filelist_path: str,
        segment_size: int = 12800,
        hop_length: int = 400,
        use_f0: bool = True,
        speaker_id: int = 0,
    ):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.use_f0 = use_f0
        self.speaker_id = speaker_id

        # Load file list
        self.audio_files = self._load_filelist(filelist_path)
        logger.info(f"Loaded {len(self.audio_files)} files from {filelist_path}")

    def _load_filelist(self, filelist_path: str) -> List[str]:
        """Load file list from text file."""
        files = []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Support both full paths and relative paths
                    if os.path.isabs(line):
                        files.append(line)
                    else:
                        files.append(os.path.join(self.data_dir, line))
        return files

    def _load_spec(self, audio_path: str) -> torch.Tensor:
        """Load or compute mel spectrogram."""
        spec_path = audio_path.replace('.wav', '.spec.pt')
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            # Fallback: try .npy format
            spec_npy_path = audio_path.replace('.wav', '_spec.npy')
            if os.path.exists(spec_npy_path):
                spec = torch.from_numpy(np.load(spec_npy_path))
            else:
                raise FileNotFoundError(f"Spectrogram not found: {spec_path}")
        return spec

    def _load_phone(self, audio_path: str) -> torch.Tensor:
        """Load phone features."""
        # Try different naming conventions
        phone_paths = [
            audio_path.replace('.wav', '_phone.npy'),
            audio_path.replace('.wav', '.phone.npy'),
            os.path.join(os.path.dirname(audio_path), 'phone',
                        os.path.basename(audio_path).replace('.wav', '.npy')),
        ]

        for phone_path in phone_paths:
            if os.path.exists(phone_path):
                return torch.from_numpy(np.load(phone_path))

        raise FileNotFoundError(f"Phone features not found for: {audio_path}")

    def _load_pitch(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load pitch features (quantized and continuous)."""
        # Quantized pitch
        pitch_paths = [
            audio_path.replace('.wav', '_pitch.npy'),
            audio_path.replace('.wav', '.pitch.npy'),
        ]
        pitch = None
        for path in pitch_paths:
            if os.path.exists(path):
                pitch = torch.from_numpy(np.load(path))
                break

        # Continuous pitch (pitchf)
        pitchf_paths = [
            audio_path.replace('.wav', '_pitchf.npy'),
            audio_path.replace('.wav', '.pitchf.npy'),
        ]
        pitchf = None
        for path in pitchf_paths:
            if os.path.exists(path):
                pitchf = torch.from_numpy(np.load(path))
                break

        if pitch is None:
            raise FileNotFoundError(f"Pitch features not found for: {audio_path}")

        if pitchf is None:
            # Use pitch as pitchf if not available
            pitchf = pitch.float()

        return pitch, pitchf

    def _load_wav(self, audio_path: str) -> torch.Tensor:
        """Load audio waveform."""
        wav_pt_path = audio_path.replace('.wav', '.wav.pt')
        if os.path.exists(wav_pt_path):
            wav = torch.load(wav_pt_path)
        else:
            # Fallback: load from numpy
            wav_npy_path = audio_path.replace('.wav', '_wav.npy')
            if os.path.exists(wav_npy_path):
                wav = torch.from_numpy(np.load(wav_npy_path))
            else:
                # Try loading raw wav file
                try:
                    import librosa
                    wav, _ = librosa.load(audio_path, sr=None, mono=True)
                    wav = torch.from_numpy(wav)
                except ImportError:
                    raise FileNotFoundError(f"Waveform not found: {audio_path}")
        return wav

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dictionary containing:
            - spec: Mel spectrogram
            - wav: Audio waveform
            - phone: Phone features
            - pitch: Quantized pitch (if use_f0)
            - pitchf: Continuous pitch (if use_f0)
            - speaker_id: Speaker ID
        """
        audio_path = self.audio_files[index]

        # Load features
        spec = self._load_spec(audio_path)
        phone = self._load_phone(audio_path)
        wav = self._load_wav(audio_path)

        # Align lengths
        spec_len = spec.shape[-1]
        phone_len = phone.shape[0]
        min_len = min(spec_len, phone_len)

        spec = spec[..., :min_len]
        phone = phone[:min_len]

        # Load pitch if needed
        if self.use_f0:
            pitch, pitchf = self._load_pitch(audio_path)
            pitch = pitch[:min_len]
            pitchf = pitchf[:min_len]
        else:
            pitch = torch.zeros(min_len, dtype=torch.long)
            pitchf = torch.zeros(min_len, dtype=torch.float)

        # Align wav length
        wav_len = min_len * self.hop_length
        if wav.shape[-1] > wav_len:
            wav = wav[..., :wav_len]
        elif wav.shape[-1] < wav_len:
            wav = torch.nn.functional.pad(wav, (0, wav_len - wav.shape[-1]))

        # Random segment for training
        if min_len > self.segment_size // self.hop_length:
            max_start = min_len - self.segment_size // self.hop_length
            start = random.randint(0, max_start)
            end = start + self.segment_size // self.hop_length

            spec = spec[..., start:end]
            phone = phone[start:end]
            pitch = pitch[start:end]
            pitchf = pitchf[start:end]
            wav = wav[..., start * self.hop_length:end * self.hop_length]

        return {
            'spec': spec,
            'wav': wav,
            'phone': phone,
            'pitch': pitch,
            'pitchf': pitchf,
            'speaker_id': torch.tensor(self.speaker_id, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.audio_files)


class LoRATrainingCollate:
    """Collate function for LoRA training batches.

    Pads sequences to the same length within a batch.

    Args:
        use_f0: Whether F0 features are used
    """

    def __init__(self, use_f0: bool = True):
        self.use_f0 = use_f0

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary with padded tensors
        """
        # Get max lengths
        max_spec_len = max(sample['spec'].shape[-1] for sample in batch)
        max_wav_len = max(sample['wav'].shape[-1] for sample in batch)

        # Initialize batch tensors
        batch_size = len(batch)
        spec_channels = batch[0]['spec'].shape[0] if batch[0]['spec'].dim() > 1 else 1

        specs = torch.zeros(batch_size, spec_channels, max_spec_len)
        wavs = torch.zeros(batch_size, max_wav_len)
        phones = torch.zeros(batch_size, max_spec_len, batch[0]['phone'].shape[-1] if batch[0]['phone'].dim() > 1 else 256)
        pitches = torch.zeros(batch_size, max_spec_len, dtype=torch.long)
        pitchfs = torch.zeros(batch_size, max_spec_len)
        speaker_ids = torch.zeros(batch_size, dtype=torch.long)
        spec_lengths = torch.zeros(batch_size, dtype=torch.long)
        wav_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, sample in enumerate(batch):
            spec = sample['spec']
            wav = sample['wav']
            phone = sample['phone']
            pitch = sample['pitch']
            pitchf = sample['pitchf']

            spec_len = spec.shape[-1]
            wav_len = wav.shape[-1]

            if spec.dim() == 1:
                spec = spec.unsqueeze(0)

            specs[i, :, :spec_len] = spec
            wavs[i, :wav_len] = wav
            spec_lengths[i] = spec_len
            wav_lengths[i] = wav_len

            if phone.dim() == 1:
                phones[i, :spec_len, :phone.shape[0]] = phone.unsqueeze(0).expand(spec_len, -1)
            else:
                phones[i, :spec_len, :phone.shape[-1]] = phone[:spec_len]

            pitches[i, :spec_len] = pitch[:spec_len]
            pitchfs[i, :spec_len] = pitchf[:spec_len]
            speaker_ids[i] = sample['speaker_id']

        return {
            'spec': specs,
            'wav': wavs,
            'phone': phones,
            'pitch': pitches,
            'pitchf': pitchfs,
            'speaker_id': speaker_ids,
            'spec_lengths': spec_lengths,
            'wav_lengths': wav_lengths,
        }


class SimpleAudioDataset(Dataset):
    """Simple dataset for basic LoRA fine-tuning.

    Loads pre-processed audio features from a directory structure:
    data_dir/
        audio1.wav
        audio1.spec.pt
        audio1_phone.npy
        audio1_pitch.npy (optional)
        ...

    Args:
        data_dir: Directory containing processed audio files
        segment_size: Size of audio segments
        hop_length: Hop length for feature extraction
        use_f0: Whether to use F0 features
    """

    def __init__(
        self,
        data_dir: str,
        segment_size: int = 12800,
        hop_length: int = 400,
        use_f0: bool = True,
    ):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.use_f0 = use_f0

        # Find all spec files
        self.files = []
        for f in os.listdir(data_dir):
            if f.endswith('.spec.pt'):
                base_name = f.replace('.spec.pt', '')
                self.files.append(base_name)

        logger.info(f"Found {len(self.files)} samples in {data_dir}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base_name = self.files[index]
        base_path = os.path.join(self.data_dir, base_name)

        # Load spec
        spec = torch.load(f"{base_path}.spec.pt")

        # Load phone
        phone = torch.from_numpy(np.load(f"{base_path}_phone.npy"))

        # Load wav
        if os.path.exists(f"{base_path}.wav.pt"):
            wav = torch.load(f"{base_path}.wav.pt")
        else:
            wav = torch.zeros(spec.shape[-1] * self.hop_length)

        # Load pitch
        if self.use_f0 and os.path.exists(f"{base_path}_pitch.npy"):
            pitch = torch.from_numpy(np.load(f"{base_path}_pitch.npy")).long()
            if os.path.exists(f"{base_path}_pitchf.npy"):
                pitchf = torch.from_numpy(np.load(f"{base_path}_pitchf.npy")).float()
            else:
                pitchf = pitch.float()
        else:
            pitch = torch.zeros(spec.shape[-1], dtype=torch.long)
            pitchf = torch.zeros(spec.shape[-1], dtype=torch.float)

        # Align lengths
        min_len = min(spec.shape[-1], phone.shape[0], pitch.shape[0])
        spec = spec[..., :min_len]
        phone = phone[:min_len]
        pitch = pitch[:min_len]
        pitchf = pitchf[:min_len]

        return {
            'spec': spec,
            'wav': wav,
            'phone': phone,
            'pitch': pitch,
            'pitchf': pitchf,
            'speaker_id': torch.tensor(0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.files)


class PreprocessedDataset(Dataset):
    """Dataset for loading preprocessed .pt files from PreprocessingPipeline.

    This dataset loads consolidated feature files created by the preprocessing
    pipeline, which contain all features in a single .pt file.

    Args:
        data_dir: Directory containing .pt files (training_data folder)
        filelist_path: Path to filelist.txt (optional, if None loads all .pt files)
        segment_size: Size of audio segments for training
        hop_length: Hop length used for feature extraction
        use_f0: Whether to use F0 (pitch) features
        speaker_id: Speaker ID for single-speaker training
    """

    def __init__(
        self,
        data_dir: str,
        filelist_path: Optional[str] = None,
        segment_size: int = 12800,
        hop_length: int = 400,
        use_f0: bool = True,
        speaker_id: int = 0,
    ):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.use_f0 = use_f0
        self.speaker_id = speaker_id

        # Load file list
        if filelist_path is not None and os.path.exists(filelist_path):
            self.files = self._load_filelist(filelist_path)
        else:
            # Auto-discover .pt files
            self.files = self._discover_files()

        logger.info(f"PreprocessedDataset: Found {len(self.files)} samples")

    def _load_filelist(self, filelist_path: str) -> List[str]:
        """Load file list from text file."""
        files = []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    files.append(line)
        return files

    def _discover_files(self) -> List[str]:
        """Discover all .pt files in data directory, filtering out short samples."""
        files = []
        skipped = 0
        min_frames = 32  # Minimum frames required for RVC

        for f in os.listdir(self.data_dir):
            if f.endswith('.pt'):
                # Check if sample is long enough
                pt_path = os.path.join(self.data_dir, f)
                try:
                    data = torch.load(pt_path, weights_only=False)
                    phone_len = data['phone'].shape[0]
                    if phone_len >= min_frames:
                        files.append(f.replace('.pt', ''))
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
                    skipped += 1

        if skipped > 0:
            logger.info(f"Skipped {skipped} samples with < {min_frames} frames")

        return sorted(files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        base_name = self.files[index]
        pt_path = os.path.join(self.data_dir, f"{base_name}.pt")

        # Load consolidated features
        data = torch.load(pt_path)

        phone = data['phone']
        pitch = data['pitch']
        pitchf = data['pitchf']
        spec = data['spec']
        wav = data['wav']

        # Ensure correct types
        if not isinstance(phone, torch.Tensor):
            phone = torch.from_numpy(phone)
        if not isinstance(pitch, torch.Tensor):
            pitch = torch.from_numpy(pitch)
        if not isinstance(pitchf, torch.Tensor):
            pitchf = torch.from_numpy(pitchf)
        if not isinstance(spec, torch.Tensor):
            spec = torch.from_numpy(spec)
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)

        pitch = pitch.long()
        pitchf = pitchf.float()

        # Get minimum length across features
        min_len = min(phone.shape[0], pitch.shape[0], spec.shape[-1])

        # Align all features
        phone = phone[:min_len]
        pitch = pitch[:min_len]
        pitchf = pitchf[:min_len]
        spec = spec[..., :min_len]

        # Align wav length
        wav_len = min_len * self.hop_length
        if wav.shape[-1] > wav_len:
            wav = wav[..., :wav_len]
        elif wav.shape[-1] < wav_len:
            wav = torch.nn.functional.pad(wav, (0, wav_len - wav.shape[-1]))

        # Random segment for training
        segment_frames = self.segment_size // self.hop_length
        if min_len > segment_frames:
            max_start = min_len - segment_frames
            start = random.randint(0, max_start)
            end = start + segment_frames

            spec = spec[..., start:end]
            phone = phone[start:end]
            pitch = pitch[start:end]
            pitchf = pitchf[start:end]
            wav = wav[..., start * self.hop_length:end * self.hop_length]

        if not self.use_f0:
            pitch = torch.zeros_like(pitch)
            pitchf = torch.zeros_like(pitchf)

        return {
            'spec': spec,
            'wav': wav,
            'phone': phone,
            'pitch': pitch,
            'pitchf': pitchf,
            'speaker_id': torch.tensor(self.speaker_id, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.files)


def create_dataloader(
    data_dir: str,
    filelist_path: Optional[str] = None,
    batch_size: int = 4,
    segment_size: int = 12800,
    hop_length: int = 400,
    use_f0: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    dataset_type: str = "auto",
) -> DataLoader:
    """Create a DataLoader for LoRA training.

    Args:
        data_dir: Directory containing training data
        filelist_path: Path to file list (optional)
        batch_size: Batch size
        segment_size: Audio segment size
        hop_length: Hop length
        use_f0: Whether to use F0 features
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data
        dataset_type: Dataset type - "auto", "preprocessed", "simple", or "filelist"

    Returns:
        DataLoader instance
    """
    # Auto-detect dataset type
    if dataset_type == "auto":
        # Check for preprocessed .pt files
        training_data_dir = os.path.join(data_dir, "training_data")
        if os.path.exists(training_data_dir):
            pt_files = [f for f in os.listdir(training_data_dir) if f.endswith('.pt')]
            if pt_files:
                dataset_type = "preprocessed"
                data_dir = training_data_dir
                logger.info(f"Auto-detected preprocessed dataset in {training_data_dir}")

        if dataset_type == "auto":
            # Check for .pt files in data_dir directly
            pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
            if pt_files:
                dataset_type = "preprocessed"
                logger.info(f"Auto-detected preprocessed dataset in {data_dir}")

        if dataset_type == "auto":
            # Fall back to other types
            if filelist_path is not None:
                dataset_type = "filelist"
            else:
                dataset_type = "simple"

    # Create dataset based on type
    if dataset_type == "preprocessed":
        dataset = PreprocessedDataset(
            data_dir=data_dir,
            filelist_path=filelist_path,
            segment_size=segment_size,
            hop_length=hop_length,
            use_f0=use_f0,
        )
    elif dataset_type == "filelist" and filelist_path is not None:
        dataset = LoRATrainingDataset(
            data_dir=data_dir,
            filelist_path=filelist_path,
            segment_size=segment_size,
            hop_length=hop_length,
            use_f0=use_f0,
        )
    else:
        dataset = SimpleAudioDataset(
            data_dir=data_dir,
            segment_size=segment_size,
            hop_length=hop_length,
            use_f0=use_f0,
        )

    collate_fn = LoRATrainingCollate(use_f0=use_f0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader
