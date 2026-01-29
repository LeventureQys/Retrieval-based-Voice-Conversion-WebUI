"""
Prepare long audio for training: slice first, then denoise with DeepFilter.
This approach saves memory by processing smaller segments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def slice_audio(
    input_path: str,
    output_dir: str,
    segment_length: float = 3.0,
    min_length: float = 1.5,  # 增加最小长度，确保有足够的帧数
    silence_threshold: float = -40.0,
    target_sr: int = 40000,
) -> list:
    """Slice audio into segments based on silence detection.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save segments
        segment_length: Target segment length in seconds
        min_length: Minimum segment length in seconds (should be >= 1.5s for RVC)
        silence_threshold: Silence threshold in dB
        target_sr: Target sample rate

    Returns:
        List of output file paths
    """
    import librosa
    import soundfile as sf

    logger.info(f"Loading audio for slicing: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    logger.info(f"Audio loaded: {len(audio)/sr:.2f} seconds at {sr} Hz")

    # Convert to target sample rate if needed
    if sr != target_sr:
        logger.info(f"Resampling from {sr} Hz to {target_sr} Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Detect non-silent intervals
    logger.info("Detecting voice segments...")
    intervals = librosa.effects.split(
        audio,
        top_db=-silence_threshold,
        frame_length=int(sr * 0.025),  # 25ms frames
        hop_length=int(sr * 0.010),    # 10ms hop
    )

    logger.info(f"Found {len(intervals)} voice segments")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process segments
    output_files = []
    segment_idx = 0
    skipped_count = 0

    target_samples = int(segment_length * sr)
    min_samples = int(min_length * sr)

    current_segment = []
    current_length = 0

    for start, end in intervals:
        segment = audio[start:end]
        segment_len = len(segment)

        # If adding this segment would exceed target length, save current and start new
        if current_length + segment_len > target_samples * 1.5 and current_length >= min_samples:
            # Save current segment
            combined = np.concatenate(current_segment)
            output_path = os.path.join(output_dir, f"segment_{segment_idx:04d}.wav")
            sf.write(output_path, combined, sr)
            output_files.append(output_path)
            segment_idx += 1

            current_segment = []
            current_length = 0

        current_segment.append(segment)
        current_length += segment_len

        # Add small silence between segments
        silence = np.zeros(int(0.1 * sr))
        current_segment.append(silence)
        current_length += len(silence)

    # Save remaining segment only if it's long enough
    if current_length >= min_samples:
        combined = np.concatenate(current_segment)
        output_path = os.path.join(output_dir, f"segment_{segment_idx:04d}.wav")
        sf.write(output_path, combined, sr)
        output_files.append(output_path)
    else:
        skipped_count += 1

    logger.info(f"Created {len(output_files)} segments (skipped {skipped_count} short segments)")

    return output_files


def denoise_segments(input_dir: str, output_dir: str, target_sr: int = 40000) -> list:
    """Denoise all audio segments in a directory using DeepFilterNet.

    Args:
        input_dir: Directory containing audio segments
        output_dir: Directory to save denoised segments
        target_sr: Target sample rate for output files

    Returns:
        List of denoised file paths
    """
    from df.enhance import enhance, init_df, load_audio, save_audio
    import glob
    import librosa
    import soundfile as sf

    logger.info(f"Loading DeepFilter model...")
    model, df_state, _ = init_df()

    os.makedirs(output_dir, exist_ok=True)

    # Find all wav files
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    logger.info(f"Found {len(input_files)} segments to denoise")

    output_files = []
    for i, input_path in enumerate(input_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            logger.info(f"[{i+1}/{len(input_files)}] Skipping {filename} (already exists)")
            output_files.append(output_path)
            continue

        logger.info(f"[{i+1}/{len(input_files)}] Denoising {filename}...")

        try:
            audio, sr = load_audio(input_path, sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)

            # Convert to numpy and resample to target sample rate
            enhanced_np = enhanced.squeeze().cpu().numpy()
            if sr != target_sr:
                enhanced_np = librosa.resample(enhanced_np, orig_sr=sr, target_sr=target_sr)

            # Save with target sample rate
            sf.write(output_path, enhanced_np, target_sr)
            output_files.append(output_path)
        except Exception as e:
            logger.error(f"Failed to denoise {filename}: {e}")
            # Copy original file if denoising fails
            import shutil
            shutil.copy(input_path, output_path)
            output_files.append(output_path)

    logger.info(f"Denoised {len(output_files)} segments")
    return output_files


def main():
    parser = argparse.ArgumentParser(description='Prepare long audio for training')
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    parser.add_argument('--skip_denoise', action='store_true', help='Skip denoising step')
    parser.add_argument('--segment_length', type=float, default=3.0, help='Target segment length in seconds')
    parser.add_argument('--min_length', type=float, default=1.0, help='Minimum segment length')
    parser.add_argument('--target_sr', type=int, default=40000, help='Target sample rate')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Slice audio first (saves memory)
    logger.info("=" * 60)
    logger.info("Step 1: Slicing audio into segments")
    logger.info("=" * 60)

    raw_segments_dir = os.path.join(args.output_dir, "raw_segments")
    segment_files = slice_audio(
        args.input,
        raw_segments_dir,
        segment_length=args.segment_length,
        min_length=args.min_length,
        target_sr=args.target_sr,
    )

    # Step 2: Denoise each segment
    if not args.skip_denoise:
        logger.info("=" * 60)
        logger.info("Step 2: Denoising segments with DeepFilter")
        logger.info("=" * 60)

        denoised_dir = os.path.join(args.output_dir, "denoised_segments")
        output_files = denoise_segments(raw_segments_dir, denoised_dir)
        final_dir = denoised_dir
    else:
        output_files = segment_files
        final_dir = raw_segments_dir

    # Summary
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total segments: {len(output_files)}")
    logger.info(f"Output directory: {final_dir}")
    logger.info("=" * 60)

    return output_files


if __name__ == "__main__":
    main()
