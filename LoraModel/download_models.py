#!/usr/bin/env python
"""
Model Download Script for RVC-LoRA

Downloads required pretrained models from HuggingFace:
1. HuBERT base model (for feature extraction)
2. RMVPE model (for F0 pitch extraction)
3. Pretrained RVC v2 models (base models for LoRA training)

Usage:
    python download_models.py --all                    # Download all models
    python download_models.py --hubert                 # Download HuBERT only
    python download_models.py --rmvpe                  # Download RMVPE only
    python download_models.py --pretrained_v2          # Download pretrained_v2 models only
    python download_models.py --pretrained_v2 --models f0G40k.pth  # Download specific model
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

# HuggingFace repository
HF_REPO = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

# Get script directory
SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = SCRIPT_DIR / "download"

# Model definitions
MODELS = {
    "hubert": {
        "url": HF_REPO + "hubert_base.pt",
        "path": DOWNLOAD_DIR / "hubert_base.pt",
        "size": "189 MB",
        "description": "HuBERT base model for feature extraction"
    },
    "rmvpe": {
        "url": HF_REPO + "rmvpe.pt",
        "path": DOWNLOAD_DIR / "rmvpe.pt",
        "size": "55 MB",
        "description": "RMVPE model for F0 pitch extraction"
    },
}

# Pretrained v2 models
PRETRAINED_V2_MODELS = {
    "D32k.pth": "Discriminator for 32kHz",
    "D40k.pth": "Discriminator for 40kHz",
    "D48k.pth": "Discriminator for 48kHz",
    "G32k.pth": "Generator for 32kHz (no F0)",
    "G40k.pth": "Generator for 40kHz (no F0)",
    "G48k.pth": "Generator for 48kHz (no F0)",
    "f0D32k.pth": "F0 Discriminator for 32kHz",
    "f0D40k.pth": "F0 Discriminator for 40kHz",
    "f0D48k.pth": "F0 Discriminator for 48kHz",
    "f0G32k.pth": "F0 Generator for 32kHz",
    "f0G40k.pth": "F0 Generator for 40kHz (recommended)",
    "f0G48k.pth": "F0 Generator for 48kHz",
}


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists():
        print(f"✓ {output_path.name} already exists, skipping download")
        return True

    print(f"Downloading {output_path.name}...")
    if description:
        print(f"  Description: {description}")

    try:
        def reporthook(block_num, block_size, total_size):
            """Progress callback for urllib."""
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(downloaded * 100.0 / total_size, 100)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')

        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        print()  # New line after progress
        print(f"✓ Downloaded {output_path.name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        if output_path.exists():
            output_path.unlink()
        return False
    except urllib.error.URLError as e:
        print(f"\n✗ URL Error: {e.reason}")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        print(f"\n✗ Failed to download {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_hubert():
    """Download HuBERT model."""
    print("\n" + "="*60)
    print("Downloading HuBERT Model")
    print("="*60)
    model = MODELS["hubert"]
    return download_file(model["url"], model["path"], model["description"])


def download_rmvpe():
    """Download RMVPE model."""
    print("\n" + "="*60)
    print("Downloading RMVPE Model")
    print("="*60)
    model = MODELS["rmvpe"]
    return download_file(model["url"], model["path"], model["description"])


def download_pretrained_v2(model_names: Optional[List[str]] = None):
    """Download pretrained v2 models."""
    print("\n" + "="*60)
    print("Downloading Pretrained V2 Models")
    print("="*60)

    pretrained_dir = DOWNLOAD_DIR / "pretrained_v2"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # If no specific models specified, download all
    if not model_names:
        model_names = list(PRETRAINED_V2_MODELS.keys())

    success_count = 0
    total_count = len(model_names)

    for model_name in model_names:
        if model_name not in PRETRAINED_V2_MODELS:
            print(f"✗ Unknown model: {model_name}")
            print(f"  Available models: {', '.join(PRETRAINED_V2_MODELS.keys())}")
            continue

        url = HF_REPO + "pretrained_v2/" + model_name
        output_path = pretrained_dir / model_name
        description = PRETRAINED_V2_MODELS[model_name]

        if download_file(url, output_path, description):
            success_count += 1

    print(f"\nDownloaded {success_count}/{total_count} pretrained_v2 models")
    return success_count == total_count


def list_models():
    """List all available models."""
    print("\n" + "="*60)
    print("Available Models")
    print("="*60)

    print("\n1. HuBERT Model:")
    print(f"   - {MODELS['hubert']['description']}")
    print(f"   - Size: {MODELS['hubert']['size']}")

    print("\n2. RMVPE Model:")
    print(f"   - {MODELS['rmvpe']['description']}")
    print(f"   - Size: {MODELS['rmvpe']['size']}")

    print("\n3. Pretrained V2 Models:")
    for name, desc in PRETRAINED_V2_MODELS.items():
        status = "✓" if (DOWNLOAD_DIR / "pretrained_v2" / name).exists() else " "
        print(f"   [{status}] {name:15s} - {desc}")

    print("\n" + "="*60)


def check_downloads():
    """Check which models are already downloaded."""
    print("\n" + "="*60)
    print("Download Status")
    print("="*60)

    # Check HuBERT
    hubert_exists = MODELS["hubert"]["path"].exists()
    print(f"{'✓' if hubert_exists else '✗'} HuBERT: {MODELS['hubert']['path']}")

    # Check RMVPE
    rmvpe_exists = MODELS["rmvpe"]["path"].exists()
    print(f"{'✓' if rmvpe_exists else '✗'} RMVPE: {MODELS['rmvpe']['path']}")

    # Check pretrained_v2
    print("\nPretrained V2 Models:")
    pretrained_dir = DOWNLOAD_DIR / "pretrained_v2"
    for name in PRETRAINED_V2_MODELS.keys():
        exists = (pretrained_dir / name).exists()
        print(f"  {'✓' if exists else '✗'} {name}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for RVC-LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python download_models.py --all

  # Download specific components
  python download_models.py --hubert --rmvpe
  python download_models.py --pretrained_v2

  # Download specific pretrained_v2 models
  python download_models.py --pretrained_v2 --models f0G40k.pth f0G48k.pth

  # List available models
  python download_models.py --list

  # Check download status
  python download_models.py --check
        """
    )

    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--hubert", action="store_true", help="Download HuBERT model")
    parser.add_argument("--rmvpe", action="store_true", help="Download RMVPE model")
    parser.add_argument("--pretrained_v2", action="store_true", help="Download pretrained_v2 models")
    parser.add_argument("--models", nargs="+", help="Specific pretrained_v2 models to download")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--check", action="store_true", help="Check download status")

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # List models
    if args.list:
        list_models()
        return

    # Check downloads
    if args.check:
        check_downloads()
        return

    # Download models
    print("RVC-LoRA Model Downloader")
    print(f"Download directory: {DOWNLOAD_DIR}")

    success = True

    if args.all or args.hubert:
        if not download_hubert():
            success = False

    if args.all or args.rmvpe:
        if not download_rmvpe():
            success = False

    if args.all or args.pretrained_v2:
        if not download_pretrained_v2(args.models):
            success = False

    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ All downloads completed successfully!")
    else:
        print("✗ Some downloads failed. Please check the errors above.")
    print("="*60)

    # Show next steps
    print("\nNext steps:")
    print("1. Verify downloads: python download_models.py --check")
    print("2. Start training: python scripts/train_lora_e2e.py --input_dir <audio_dir> --output_dir <output_dir>")
    print("3. Run inference: python scripts/infer_lora_e2e.py --source <input.wav> --output <output.wav> --lora <lora.pth>")


if __name__ == "__main__":
    main()
