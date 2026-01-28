"""
Audio processing utilities for RVC-LoRA

This module handles audio loading, resampling, slicing, and normalization.
"""

import os
import logging
from typing import List, Tuple, Optional, Generator
import numpy as np
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def load_audio(file_path: str, target_sr: int) -> np.ndarray:
    """Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Audio waveform as numpy array (float32)
    """
    try:
        import ffmpeg

        file_path = _clean_path(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        out, _ = (
            ffmpeg.input(file_path, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=target_sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.float32).flatten()
    except Exception as e:
        # Fallback to librosa
        try:
            import librosa
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio.astype(np.float32)
        except ImportError:
            raise RuntimeError(f"Failed to load audio: {e}. Install ffmpeg or librosa.")


def _clean_path(path_str: str) -> str:
    """Clean file path string."""
    import platform
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


class Slicer:
    """Audio slicer for splitting audio by silence.

    This is adapted from RVC's slicer2.py for use in LoRA preprocessing.

    Args:
        sr: Sample rate
        threshold: Silence threshold in dB (default: -40)
        min_length: Minimum length of audio clip in ms (default: 5000)
        min_interval: Minimum silence interval for slicing in ms (default: 300)
        hop_size: Hop size in ms (default: 20)
        max_sil_kept: Maximum silence to keep around clips in ms (default: 5000)
    """

    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "Must satisfy: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError("Must satisfy: max_sil_kept >= hop_size")

        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _get_rms(self, y: np.ndarray) -> np.ndarray:
        """Calculate RMS energy."""
        frame_length = self.win_size
        hop_length = self.hop_size

        padding = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode="constant")

        out_strides = y.strides + tuple([y.strides[-1]])
        x_shape_trimmed = list(y.shape)
        x_shape_trimmed[-1] -= frame_length - 1
        out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
        xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
        xw = np.moveaxis(xw, -1, -2)

        slices = [slice(None)] * xw.ndim
        slices[-1] = slice(0, None, hop_length)
        x = xw[tuple(slices)]

        power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
        return np.sqrt(power)

    def _apply_slice(self, waveform: np.ndarray, begin: int, end: int) -> np.ndarray:
        """Apply slice to waveform."""
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Slice audio by silence detection.

        Args:
            waveform: Audio waveform

        Returns:
            List of audio chunks
        """
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = self._get_rms(samples).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0

        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue

            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        # Handle trailing silence
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Apply slices
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks


class AudioProcessor:
    """Audio processor for RVC-LoRA preprocessing.

    Handles audio loading, filtering, slicing, and normalization.

    Args:
        target_sr: Target sample rate (default: 40000 for RVC)
        slice_audio: Whether to slice audio by silence (default: True)
        segment_duration: Duration of each segment in seconds (default: 3.7)
        overlap: Overlap ratio between segments (default: 0.3)
    """

    def __init__(
        self,
        target_sr: int = 40000,
        slice_audio: bool = True,
        segment_duration: float = 3.7,
        overlap: float = 0.3,
    ):
        self.target_sr = target_sr
        self.slice_audio = slice_audio
        self.segment_duration = segment_duration
        self.overlap = overlap

        # High-pass filter coefficients (48 Hz cutoff)
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=target_sr)

        # Slicer for silence detection
        if slice_audio:
            self.slicer = Slicer(
                sr=target_sr,
                threshold=-42,
                min_length=1500,
                min_interval=400,
                hop_size=15,
                max_sil_kept=500,
            )

        # Normalization parameters
        self.max_amplitude = 0.9
        self.alpha = 0.75

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file at target sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Audio waveform
        """
        return load_audio(file_path, self.target_sr)

    def apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low frequency noise.

        Args:
            audio: Input audio

        Returns:
            Filtered audio
        """
        return signal.lfilter(self.bh, self.ah, audio)

    def normalize(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Normalize audio amplitude.

        Args:
            audio: Input audio

        Returns:
            Normalized audio, or None if audio is too loud (clipped)
        """
        max_val = np.abs(audio).max()

        # Filter out clipped audio
        if max_val > 2.5:
            logger.warning(f"Audio too loud (max={max_val:.2f}), skipping")
            return None

        # Normalize
        audio = (audio / max_val * (self.max_amplitude * self.alpha)) + (1 - self.alpha) * audio
        return audio.astype(np.float32)

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling using scipy
            from scipy.signal import resample
            num_samples = int(len(audio) * target_sr / orig_sr)
            return resample(audio, num_samples).astype(np.float32)

    def process_file(
        self,
        file_path: str,
        output_dir: str,
        file_index: int = 0,
    ) -> List[Tuple[str, str]]:
        """Process a single audio file.

        Args:
            file_path: Path to input audio file
            output_dir: Output directory
            file_index: Index for naming output files

        Returns:
            List of (original_sr_path, 16k_path) tuples for processed segments
        """
        # Create output directories
        gt_wavs_dir = os.path.join(output_dir, "0_gt_wavs")
        wavs16k_dir = os.path.join(output_dir, "1_16k_wavs")
        os.makedirs(gt_wavs_dir, exist_ok=True)
        os.makedirs(wavs16k_dir, exist_ok=True)

        # Load and filter audio
        audio = self.load_audio(file_path)
        audio = self.apply_highpass_filter(audio)

        output_files = []
        segment_idx = 0

        # Slice audio if enabled
        if self.slice_audio:
            chunks = self.slicer.slice(audio)
        else:
            chunks = [audio]

        # Process each chunk
        for chunk in chunks:
            # Split into segments with overlap
            segments = list(self._split_into_segments(chunk))

            for segment in segments:
                # Normalize
                normalized = self.normalize(segment)
                if normalized is None:
                    continue

                # Save at original sample rate
                gt_path = os.path.join(gt_wavs_dir, f"{file_index}_{segment_idx}.wav")
                wavfile.write(gt_path, self.target_sr, normalized)

                # Resample to 16kHz and save
                audio_16k = self.resample(normalized, self.target_sr, 16000)
                wav16k_path = os.path.join(wavs16k_dir, f"{file_index}_{segment_idx}.wav")
                wavfile.write(wav16k_path, 16000, audio_16k)

                output_files.append((gt_path, wav16k_path))
                segment_idx += 1

        return output_files

    def _split_into_segments(self, audio: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Split audio into overlapping segments.

        Args:
            audio: Input audio

        Yields:
            Audio segments
        """
        segment_samples = int(self.target_sr * self.segment_duration)
        overlap_samples = int(segment_samples * self.overlap)
        step = segment_samples - overlap_samples
        tail_samples = int(self.target_sr * (self.segment_duration + self.overlap))

        i = 0
        while True:
            start = step * i
            i += 1

            if len(audio[start:]) > tail_samples:
                yield audio[start:start + segment_samples]
            else:
                # Last segment
                yield audio[start:]
                break

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a'),
    ) -> List[Tuple[str, str]]:
        """Process all audio files in a directory.

        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for processed files
            extensions: Tuple of valid audio file extensions

        Returns:
            List of (original_sr_path, 16k_path) tuples
        """
        all_outputs = []

        # Find all audio files
        audio_files = []
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(extensions):
                audio_files.append(os.path.join(input_dir, f))

        logger.info(f"Found {len(audio_files)} audio files in {input_dir}")

        # Process each file
        for idx, file_path in enumerate(audio_files):
            try:
                outputs = self.process_file(file_path, output_dir, idx)
                all_outputs.extend(outputs)
                logger.info(f"Processed {file_path} -> {len(outputs)} segments")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        logger.info(f"Total segments: {len(all_outputs)}")
        return all_outputs
