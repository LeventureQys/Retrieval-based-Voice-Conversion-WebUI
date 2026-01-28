"""
Test script for RVC-LoRA inference components

Tests model loading, inference utilities, and merge functionality.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lora import LoRAConfig, save_lora_checkpoint
from inference.model_loader import get_model_info


def test_model_info():
    """Test get_model_info function."""
    print("\n" + "=" * 60)
    print("Testing get_model_info")
    print("=" * 60)

    # Create a mock checkpoint
    temp_path = os.path.join(tempfile.gettempdir(), 'test_model_info.pth')

    try:
        # Create mock checkpoint data
        mock_config = [
            1025,  # spec_channels
            12800,  # segment_size
            192,  # inter_channels
            192,  # hidden_channels
            768,  # filter_channels
            2,  # n_heads
            6,  # n_layers
            3,  # kernel_size
            0.0,  # p_dropout
            "1",  # resblock
            [3, 7, 11],  # resblock_kernel_sizes
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # resblock_dilation_sizes
            [10, 10, 2, 2],  # upsample_rates
            512,  # upsample_initial_channel
            [20, 20, 4, 4],  # upsample_kernel_sizes
            109,  # spk_embed_dim
            256,  # gin_channels
            40000,  # sr
        ]

        checkpoint = {
            "config": mock_config,
            "weight": {"dummy": torch.zeros(1)},
            "epoch": 100,
        }

        torch.save(checkpoint, temp_path)

        # Test get_model_info
        info = get_model_info(temp_path)

        assert "config" in info
        assert info["config"]["inter_channels"] == 192
        assert info["config"]["gin_channels"] == 256
        assert info["epoch"] == 100
        assert info["has_lora"] == False

        print(f"Model info: {info}")
        print("get_model_info test passed!")

    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors on Windows

    return True


def test_lora_checkpoint_info():
    """Test get_model_info with LoRA checkpoint."""
    print("\n" + "=" * 60)
    print("Testing LoRA Checkpoint Info")
    print("=" * 60)

    temp_path = os.path.join(tempfile.gettempdir(), 'test_lora_info.pth')

    try:
        # Create mock LoRA checkpoint
        lora_state_dict = {
            "layer1.lora_A": torch.randn(8, 256),
            "layer1.lora_B": torch.randn(512, 8),
            "layer2.lora_A": torch.randn(8, 128),
            "layer2.lora_B": torch.randn(256, 8),
        }

        lora_config = LoRAConfig(r=8, lora_alpha=16)

        checkpoint = {
            "lora_state_dict": lora_state_dict,
            "config": lora_config.to_dict(),
            "epoch": 50,
        }

        torch.save(checkpoint, temp_path)

        # Test get_model_info
        info = get_model_info(temp_path)

        assert info["has_lora"] == True
        assert info["lora_params"] == 4
        assert info["epoch"] == 50

        print(f"LoRA checkpoint info: {info}")
        print("LoRA checkpoint info test passed!")

    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors on Windows

    return True


def test_lora_inference_class():
    """Test LoRAInference class initialization."""
    print("\n" + "=" * 60)
    print("Testing LoRAInference Class")
    print("=" * 60)

    # We can't fully test without a real model, but we can test imports
    from inference.infer_lora import LoRAInference

    print("LoRAInference class imported successfully!")
    print("LoRAInference class test passed!")
    return True


def test_model_loader_class():
    """Test LoRAModelLoader class."""
    print("\n" + "=" * 60)
    print("Testing LoRAModelLoader Class")
    print("=" * 60)

    from inference.model_loader import LoRAModelLoader

    # Create loader instance
    loader = LoRAModelLoader(device="cpu", is_half=False)

    assert loader.device == "cpu"
    assert loader.is_half == False

    print("LoRAModelLoader instantiated successfully!")
    print("LoRAModelLoader class test passed!")
    return True


def test_merge_script_imports():
    """Test merge script imports."""
    print("\n" + "=" * 60)
    print("Testing Merge Script Imports")
    print("=" * 60)

    # Add scripts to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

    from scripts.merge_lora import merge_lora_to_base, extract_lora_from_finetuned

    print("Merge script functions imported successfully!")
    print("Merge script import test passed!")
    return True


def test_lora_extraction_logic():
    """Test LoRA extraction using SVD."""
    print("\n" + "=" * 60)
    print("Testing LoRA Extraction Logic")
    print("=" * 60)

    # Create mock weight difference
    out_features = 256
    in_features = 128
    rank = 8

    # Create a low-rank matrix
    true_A = torch.randn(rank, in_features)
    true_B = torch.randn(out_features, rank)
    diff = true_B @ true_A

    # Extract using SVD
    U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Reconstruct
    sqrt_S = torch.sqrt(S_r)
    lora_B = U_r * sqrt_S.unsqueeze(0)
    lora_A = sqrt_S.unsqueeze(1) * Vh_r

    reconstructed = lora_B @ lora_A

    # Check reconstruction error
    error = torch.mean(torch.abs(diff - reconstructed)).item()

    print(f"Original diff shape: {diff.shape}")
    print(f"LoRA A shape: {lora_A.shape}")
    print(f"LoRA B shape: {lora_B.shape}")
    print(f"Reconstruction error: {error:.6f}")

    assert error < 1e-5, f"Reconstruction error too high: {error}"

    print("LoRA extraction logic test passed!")
    return True


def test_inference_pipeline_mock():
    """Test inference pipeline with mock model."""
    print("\n" + "=" * 60)
    print("Testing Inference Pipeline (Mock)")
    print("=" * 60)

    # Create a mock synthesizer
    class MockSynthesizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 512)

        def infer(self, phone, phone_lengths, pitch, pitchf, sid, **kwargs):
            batch_size = phone.shape[0]
            # Generate mock audio
            audio = torch.randn(batch_size, 1, 16000)
            return audio, None, None

        def remove_weight_norm(self):
            pass

    # Create mock model
    model = MockSynthesizer()
    model.eval()

    # Create mock inputs
    batch_size = 1
    seq_len = 100
    phone = torch.randn(batch_size, seq_len, 256)
    phone_lengths = torch.tensor([seq_len])
    pitch = torch.randint(0, 256, (batch_size, seq_len))
    pitchf = torch.randn(batch_size, seq_len)
    sid = torch.tensor([0])

    # Run inference
    with torch.no_grad():
        audio, _, _ = model.infer(phone, phone_lengths, pitch, pitchf, sid)

    assert audio.shape[0] == batch_size
    assert audio.shape[2] == 16000

    print(f"Output audio shape: {audio.shape}")
    print("Inference pipeline mock test passed!")
    return True


def test_feature_extraction_format():
    """Test expected feature extraction format."""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction Format")
    print("=" * 60)

    # Create mock extracted features
    seq_len = 100
    phone_dim = 256

    features = {
        'phone': np.random.randn(seq_len, phone_dim).astype(np.float32),
        'phone_lengths': np.array([seq_len]),
        'pitch': np.random.randint(0, 256, size=seq_len).astype(np.int64),
        'pitchf': np.random.randn(seq_len).astype(np.float32),
    }

    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), 'test_features.npz')

    try:
        np.savez(temp_path, **features)

        # Load and verify
        loaded = np.load(temp_path)

        assert 'phone' in loaded
        assert 'pitch' in loaded
        assert 'pitchf' in loaded
        assert loaded['phone'].shape == (seq_len, phone_dim)

        print(f"Phone shape: {loaded['phone'].shape}")
        print(f"Pitch shape: {loaded['pitch'].shape}")
        print(f"Pitchf shape: {loaded['pitchf'].shape}")

        loaded.close()  # Close the file before deleting
        print("Feature extraction format test passed!")

    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors on Windows

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RVC-LoRA Inference Component Tests")
    print("=" * 60)

    tests = [
        ("Model Info", test_model_info),
        ("LoRA Checkpoint Info", test_lora_checkpoint_info),
        ("LoRAInference Class", test_lora_inference_class),
        ("LoRAModelLoader Class", test_model_loader_class),
        ("Merge Script Imports", test_merge_script_imports),
        ("LoRA Extraction Logic", test_lora_extraction_logic),
        ("Inference Pipeline Mock", test_inference_pipeline_mock),
        ("Feature Extraction Format", test_feature_extraction_format),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\nFAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
