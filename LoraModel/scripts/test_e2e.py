#!/usr/bin/env python
"""
End-to-End Test Script for RVC-LoRA

This script tests the complete LoRA training and inference pipeline:
1. Preprocess training data (base_voice)
2. Train LoRA model
3. Run inference on test audio (test_voice)
4. Evaluate quality metrics

Metrics:
- Mel Cepstral Distortion (MCD): Lower is better, measures spectral similarity
- F0 Correlation: Higher is better, measures pitch tracking accuracy
- Signal-to-Noise Ratio (SNR): Higher is better
"""

import os
import sys
import time
import logging
import numpy as np
from scipy.io import wavfile
from typing import Tuple, Dict, Any

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_ROOT = os.path.dirname(SCRIPT_DIR)
RVC_ROOT = os.path.dirname(LORA_ROOT)

sys.path.insert(0, LORA_ROOT)
sys.path.insert(0, RVC_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """Load audio file."""
    sr, audio = wavfile.read(path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def compute_mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    """Compute mel spectrogram."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=2048, hop_length=512, win_length=2048
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_mcd(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int) -> float:
    """Compute Mel Cepstral Distortion (MCD).

    Lower is better. Typical values:
    - < 4.0: Excellent quality
    - 4.0 - 6.0: Good quality
    - 6.0 - 8.0: Acceptable quality
    - > 8.0: Poor quality
    """
    import librosa

    # Align lengths
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio = ref_audio[:min_len]
    gen_audio = gen_audio[:min_len]

    # Compute MFCCs
    n_mfcc = 13
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)

    # Align frame lengths
    min_frames = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_frames]
    gen_mfcc = gen_mfcc[:, :min_frames]

    # Compute MCD (excluding c0)
    diff = ref_mfcc[1:, :] - gen_mfcc[1:, :]
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))

    # Convert to dB scale
    mcd_db = (10.0 / np.log(10.0)) * mcd

    return mcd_db


def compute_f0_correlation(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int) -> float:
    """Compute F0 (pitch) correlation.

    Higher is better. Range: -1 to 1
    - > 0.9: Excellent pitch tracking
    - 0.7 - 0.9: Good pitch tracking
    - < 0.7: Poor pitch tracking
    """
    import librosa

    # Extract F0 using pyin
    ref_f0, _, _ = librosa.pyin(ref_audio, fmin=50, fmax=500, sr=sr)
    gen_f0, _, _ = librosa.pyin(gen_audio, fmin=50, fmax=500, sr=sr)

    # Align lengths
    min_len = min(len(ref_f0), len(gen_f0))
    ref_f0 = ref_f0[:min_len]
    gen_f0 = gen_f0[:min_len]

    # Remove NaN values (unvoiced frames)
    valid_mask = ~(np.isnan(ref_f0) | np.isnan(gen_f0))
    if valid_mask.sum() < 10:
        return 0.0

    ref_f0_valid = ref_f0[valid_mask]
    gen_f0_valid = gen_f0[valid_mask]

    # Compute correlation
    correlation = np.corrcoef(ref_f0_valid, gen_f0_valid)[0, 1]

    return correlation if not np.isnan(correlation) else 0.0


def compute_snr(ref_audio: np.ndarray, gen_audio: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio.

    Higher is better. Typical values:
    - > 20 dB: Excellent
    - 10-20 dB: Good
    - < 10 dB: Poor
    """
    # Align lengths
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio = ref_audio[:min_len]
    gen_audio = gen_audio[:min_len]

    # Compute SNR
    signal_power = np.mean(ref_audio ** 2)
    noise_power = np.mean((ref_audio - gen_audio) ** 2)

    if noise_power < 1e-10:
        return 100.0  # Perfect match

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_spectral_convergence(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int) -> float:
    """Compute spectral convergence.

    Lower is better. Range: 0 to inf
    - < 0.2: Excellent
    - 0.2 - 0.5: Good
    - > 0.5: Poor
    """
    ref_mel = compute_mel_spectrogram(ref_audio, sr)
    gen_mel = compute_mel_spectrogram(gen_audio, sr)

    # Align lengths
    min_frames = min(ref_mel.shape[1], gen_mel.shape[1])
    ref_mel = ref_mel[:, :min_frames]
    gen_mel = gen_mel[:, :min_frames]

    # Compute spectral convergence
    sc = np.linalg.norm(ref_mel - gen_mel, 'fro') / np.linalg.norm(ref_mel, 'fro')

    return sc


def evaluate_conversion(ref_path: str, gen_path: str, target_sr: int = 16000) -> Dict[str, float]:
    """Evaluate voice conversion quality."""
    logger.info(f"Evaluating: {os.path.basename(gen_path)}")

    # Load audio
    ref_audio, ref_sr = load_audio(ref_path, target_sr)
    gen_audio, gen_sr = load_audio(gen_path, target_sr)

    metrics = {}

    # Compute metrics
    try:
        metrics['mcd'] = compute_mcd(ref_audio, gen_audio, target_sr)
        logger.info(f"  MCD: {metrics['mcd']:.2f} dB")
    except Exception as e:
        logger.warning(f"  MCD computation failed: {e}")
        metrics['mcd'] = float('nan')

    try:
        metrics['f0_corr'] = compute_f0_correlation(ref_audio, gen_audio, target_sr)
        logger.info(f"  F0 Correlation: {metrics['f0_corr']:.3f}")
    except Exception as e:
        logger.warning(f"  F0 correlation failed: {e}")
        metrics['f0_corr'] = float('nan')

    try:
        metrics['spectral_convergence'] = compute_spectral_convergence(ref_audio, gen_audio, target_sr)
        logger.info(f"  Spectral Convergence: {metrics['spectral_convergence']:.3f}")
    except Exception as e:
        logger.warning(f"  Spectral convergence failed: {e}")
        metrics['spectral_convergence'] = float('nan')

    return metrics


def run_test(
    base_voice_dir: str,
    test_voice_dir: str,
    output_dir: str,
    base_model_path: str,
    epochs: int = 20,
    batch_size: int = 2,
    lora_rank: int = 8,
    skip_training: bool = False,
) -> Dict[str, Any]:
    """Run complete test pipeline."""

    results = {
        'training': {},
        'inference': {},
        'metrics': {},
    }

    # Paths
    preprocess_dir = os.path.join(output_dir, "preprocessed")
    lora_path = os.path.join(output_dir, "lora_final.pth")
    converted_dir = os.path.join(output_dir, "converted")
    os.makedirs(converted_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Training
    # =========================================================================
    if not skip_training or not os.path.exists(lora_path):
        logger.info("=" * 60)
        logger.info("Step 1: Training LoRA")
        logger.info("=" * 60)

        from scripts.train_lora_e2e import train_lora_e2e

        train_start = time.time()

        try:
            train_result = train_lora_e2e(
                input_dir=base_voice_dir,
                output_dir=output_dir,
                base_model=base_model_path,
                sample_rate=40000,
                version="v2",
                epochs=epochs,
                batch_size=batch_size,
                lora_rank=lora_rank,
                lora_alpha=lora_rank * 2,
                f0=True,
                device="cuda",
                is_half=True,
            )

            results['training'] = {
                'success': True,
                'time': time.time() - train_start,
                'epochs': epochs,
                'lora_path': lora_path,
            }
            logger.info(f"Training completed in {results['training']['time']:.1f}s")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            results['training'] = {'success': False, 'error': str(e)}
            return results
    else:
        logger.info("Skipping training, using existing LoRA weights")
        results['training'] = {'success': True, 'skipped': True, 'lora_path': lora_path}

    # =========================================================================
    # Step 2: Inference
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Running Inference")
    logger.info("=" * 60)

    from scripts.infer_lora_e2e import LoRAVoiceConverter

    try:
        converter = LoRAVoiceConverter(
            base_model_path=base_model_path,
            lora_path=lora_path,
            device="cuda",
            is_half=True,
            version="v2",
            f0=True,
            sample_rate=40000,
        )

        # Convert test files
        test_files = [f for f in os.listdir(test_voice_dir) if f.endswith('.wav')]
        converted_files = []

        for test_file in test_files:
            test_path = os.path.join(test_voice_dir, test_file)
            output_path = os.path.join(converted_dir, f"converted_{test_file}")

            logger.info(f"Converting: {test_file}")
            infer_start = time.time()

            try:
                converter.convert_file(
                    source_path=test_path,
                    output_path=output_path,
                    f0_up_key=0,
                    speaker_id=0,
                    protect=0.33,
                )

                converted_files.append({
                    'source': test_path,
                    'output': output_path,
                    'time': time.time() - infer_start,
                })
                logger.info(f"  Saved to: {output_path}")

            except Exception as e:
                logger.error(f"  Conversion failed: {e}")

        results['inference'] = {
            'success': True,
            'converted_files': converted_files,
        }

    except Exception as e:
        logger.error(f"Inference setup failed: {e}")
        results['inference'] = {'success': False, 'error': str(e)}
        return results

    # =========================================================================
    # Step 3: Evaluation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Evaluating Quality")
    logger.info("=" * 60)

    all_metrics = []

    for conv_info in converted_files:
        try:
            metrics = evaluate_conversion(
                ref_path=conv_info['source'],
                gen_path=conv_info['output'],
                target_sr=16000,
            )
            metrics['file'] = os.path.basename(conv_info['source'])
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Evaluation failed for {conv_info['source']}: {e}")

    # Compute average metrics
    if all_metrics:
        avg_metrics = {}
        for key in ['mcd', 'f0_corr', 'spectral_convergence']:
            values = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
            if values:
                avg_metrics[key] = np.mean(values)

        results['metrics'] = {
            'per_file': all_metrics,
            'average': avg_metrics,
        }

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    if results['training'].get('success'):
        logger.info(f"Training: SUCCESS")
        if not results['training'].get('skipped'):
            logger.info(f"  Time: {results['training'].get('time', 0):.1f}s")
    else:
        logger.info(f"Training: FAILED - {results['training'].get('error')}")

    if results['inference'].get('success'):
        logger.info(f"Inference: SUCCESS")
        logger.info(f"  Converted files: {len(converted_files)}")
    else:
        logger.info(f"Inference: FAILED - {results['inference'].get('error')}")

    if results['metrics'].get('average'):
        logger.info("Quality Metrics (Average):")
        avg = results['metrics']['average']
        if 'mcd' in avg:
            quality = "Excellent" if avg['mcd'] < 4 else "Good" if avg['mcd'] < 6 else "Acceptable" if avg['mcd'] < 8 else "Poor"
            logger.info(f"  MCD: {avg['mcd']:.2f} dB ({quality})")
        if 'f0_corr' in avg:
            quality = "Excellent" if avg['f0_corr'] > 0.9 else "Good" if avg['f0_corr'] > 0.7 else "Poor"
            logger.info(f"  F0 Correlation: {avg['f0_corr']:.3f} ({quality})")
        if 'spectral_convergence' in avg:
            quality = "Excellent" if avg['spectral_convergence'] < 0.2 else "Good" if avg['spectral_convergence'] < 0.5 else "Poor"
            logger.info(f"  Spectral Convergence: {avg['spectral_convergence']:.3f} ({quality})")

    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RVC-LoRA Pipeline")
    parser.add_argument("--base_voice", type=str,
                        default=os.path.join(LORA_ROOT, "download", "base_voice"),
                        help="Directory with training audio")
    parser.add_argument("--test_voice", type=str,
                        default=os.path.join(LORA_ROOT, "download", "test_voice"),
                        help="Directory with test audio")
    parser.add_argument("--output", type=str,
                        default=os.path.join(LORA_ROOT, "test_output"),
                        help="Output directory")
    parser.add_argument("--model", type=str,
                        default=os.path.join(LORA_ROOT, "download", "pretrained_v2", "f0G40k.pth"),
                        help="Base model path")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training if LoRA exists")

    args = parser.parse_args()

    results = run_test(
        base_voice_dir=args.base_voice,
        test_voice_dir=args.test_voice,
        output_dir=args.output,
        base_model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        skip_training=args.skip_training,
    )

    # Save results
    import json
    results_path = os.path.join(args.output, "test_results.json")

    # Convert non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(results_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    logger.info(f"Results saved to: {results_path}")
