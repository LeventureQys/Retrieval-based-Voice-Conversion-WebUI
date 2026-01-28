"""
Simple test script for LoRA core functionality

This script tests the basic LoRA layers without requiring the full RVC model.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lora import (
    LoRAConfig,
    LoRALinear,
    LoRAConv1d,
    LoRAConvTranspose1d,
    inject_lora,
    extract_lora_weights,
    merge_lora_weights,
    print_lora_info,
    count_lora_parameters,
)


def test_lora_linear():
    """Test LoRALinear layer."""
    print("\n" + "="*60)
    print("Testing LoRALinear")
    print("="*60)

    # Create layer
    layer = LoRALinear(128, 256, r=8, lora_alpha=16)

    # Check shapes
    assert layer.lora_A.shape == (8, 128), f"Expected (8, 128), got {layer.lora_A.shape}"
    assert layer.lora_B.shape == (256, 8), f"Expected (256, 8), got {layer.lora_B.shape}"

    # Check initialization
    assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B)), "lora_B should be zero-initialized"

    # Test forward pass
    x = torch.randn(4, 128)
    output = layer(x)
    assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"

    # Test that base weights are frozen
    assert not layer.weight.requires_grad, "Base weights should be frozen"
    assert layer.lora_A.requires_grad, "LoRA A should be trainable"
    assert layer.lora_B.requires_grad, "LoRA B should be trainable"

    print("[PASS] LoRALinear tests passed!")
    return True


def test_lora_conv1d():
    """Test LoRAConv1d layer."""
    print("\n" + "="*60)
    print("Testing LoRAConv1d")
    print("="*60)

    # Create layer
    layer = LoRAConv1d(64, 128, kernel_size=3, padding=1, r=8, lora_alpha=16)

    # Check shapes
    assert layer.lora_A.shape == (8, 64 * 3), f"Expected (8, 192), got {layer.lora_A.shape}"
    assert layer.lora_B.shape == (128, 8), f"Expected (128, 8), got {layer.lora_B.shape}"

    # Test forward pass
    x = torch.randn(4, 64, 100)
    output = layer(x)
    assert output.shape == (4, 128, 100), f"Expected (4, 128, 100), got {output.shape}"

    print("[PASS] LoRAConv1d tests passed!")
    return True


def test_lora_conv_transpose1d():
    """Test LoRAConvTranspose1d layer."""
    print("\n" + "="*60)
    print("Testing LoRAConvTranspose1d")
    print("="*60)

    # Create layer
    layer = LoRAConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, r=8, lora_alpha=16)

    # Check shapes
    assert layer.lora_A.shape == (8, 128 * 4), f"Expected (8, 512), got {layer.lora_A.shape}"
    assert layer.lora_B.shape == (64, 8), f"Expected (64, 8), got {layer.lora_B.shape}"

    # Test forward pass
    x = torch.randn(4, 128, 50)
    output = layer(x)
    assert output.shape == (4, 64, 100), f"Expected (4, 64, 100), got {output.shape}"

    print("[PASS] LoRAConvTranspose1d tests passed!")
    return True


def test_lora_injection():
    """Test LoRA injection into a simple model."""
    print("\n" + "="*60)
    print("Testing LoRA Injection")
    print("="*60)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256)
            self.conv1 = nn.Conv1d(64, 128, 3, padding=1)
            self.ups = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)

        def forward(self, x):
            return x

    model = SimpleModel()

    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")

    # Create config
    config = LoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["linear", "conv", "ups"]
    )

    # Inject LoRA
    model = inject_lora(model, config)

    # Count LoRA parameters
    lora_params, total_params = count_lora_parameters(model)
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA percentage: {lora_params/total_params*100:.2f}%")

    # Check that LoRA was injected
    assert lora_params > 0, "LoRA parameters should be > 0"
    assert lora_params < original_params, "LoRA parameters should be < original parameters"

    print("[PASS] LoRA injection tests passed!")
    return True


def test_weight_merging():
    """Test weight merging."""
    print("\n" + "="*60)
    print("Testing Weight Merging")
    print("="*60)

    # Create layer
    layer = LoRALinear(128, 256, r=8, lora_alpha=16, merge_weights=True)

    # Initially not merged
    assert not layer.merged, "Should not be merged initially"

    # Set to eval mode (should merge)
    layer.eval()
    assert layer.merged, "Should be merged in eval mode"

    # Set to train mode (should unmerge)
    layer.train()
    assert not layer.merged, "Should be unmerged in train mode"

    print("[PASS] Weight merging tests passed!")
    return True


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\n" + "="*60)
    print("Testing Checkpoint Save/Load")
    print("="*60)

    from lora import save_lora_checkpoint, load_lora_checkpoint
    import os

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 256)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    # Inject LoRA
    config = LoRAConfig(r=8, lora_alpha=16, target_modules=["linear"])
    model = inject_lora(model, config)

    # Save checkpoint
    checkpoint_path = "test_checkpoint.pth"
    save_lora_checkpoint(
        model=model,
        path=checkpoint_path,
        config=config,
        epoch=10
    )

    # Create new model
    model2 = SimpleModel()
    model2 = inject_lora(model2, config)

    # Load checkpoint
    model2, _, epoch, loaded_config = load_lora_checkpoint(
        model=model2,
        path=checkpoint_path
    )

    assert epoch == 10, f"Expected epoch 10, got {epoch}"
    assert loaded_config.r == config.r, "Config should match"

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("[PASS] Checkpoint save/load tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RVC-LoRA Core Functionality Tests")
    print("="*60)

    tests = [
        ("LoRALinear", test_lora_linear),
        ("LoRAConv1d", test_lora_conv1d),
        ("LoRAConvTranspose1d", test_lora_conv_transpose1d),
        ("LoRA Injection", test_lora_injection),
        ("Weight Merging", test_weight_merging),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
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
