"""
Test script for RVC-LoRA model integration

This script tests the GeneratorLoRA and SynthesizerLoRA classes.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lora import LoRAConfig, count_lora_parameters, print_lora_info
from models import GeneratorLoRA, SynthesizerLoRA, ResBlock1, ResBlock2


def test_resblock1():
    """Test ResBlock1 forward pass."""
    print("\n" + "=" * 60)
    print("Testing ResBlock1")
    print("=" * 60)

    # Create ResBlock1
    channels = 256
    resblock = ResBlock1(channels, kernel_size=3, dilation=(1, 3, 5))

    # Test forward pass
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, channels, seq_len)

    output = resblock(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with mask
    mask = torch.ones(batch_size, 1, seq_len)
    output_masked = resblock(x, x_mask=mask)
    assert output_masked.shape == x.shape

    # Test remove_weight_norm
    resblock.remove_weight_norm()

    print("ResBlock1 tests passed!")
    return True


def test_resblock2():
    """Test ResBlock2 forward pass."""
    print("\n" + "=" * 60)
    print("Testing ResBlock2")
    print("=" * 60)

    # Create ResBlock2
    channels = 128
    resblock = ResBlock2(channels, kernel_size=3, dilation=(1, 3))

    # Test forward pass
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, channels, seq_len)

    output = resblock(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test remove_weight_norm
    resblock.remove_weight_norm()

    print("ResBlock2 tests passed!")
    return True


def test_generator_lora_creation():
    """Test GeneratorLoRA creation without LoRA."""
    print("\n" + "=" * 60)
    print("Testing GeneratorLoRA Creation (without LoRA)")
    print("=" * 60)

    # Create GeneratorLoRA without LoRA config
    generator = GeneratorLoRA(
        initial_channel=192,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 20, 4, 4],
        gin_channels=256,
        lora_config=None,  # No LoRA
    )

    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, 192, seq_len)
    g = torch.randn(batch_size, 256, 1)

    output = generator(x, g=g)

    expected_len = seq_len * 10 * 10 * 2 * 2  # Product of upsample_rates
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output length: {expected_len}")

    assert output.shape[0] == batch_size
    assert output.shape[1] == 1  # Single channel output

    print("GeneratorLoRA creation tests passed!")
    return True


def test_generator_lora_with_lora():
    """Test GeneratorLoRA with LoRA injection."""
    print("\n" + "=" * 60)
    print("Testing GeneratorLoRA with LoRA")
    print("=" * 60)

    # Create LoRA config
    lora_config = LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["ups", "resblocks"],
    )

    # Create GeneratorLoRA with LoRA
    generator = GeneratorLoRA(
        initial_channel=192,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 20, 4, 4],
        gin_channels=256,
        lora_config=lora_config,
    )

    # Count parameters
    lora_params, total_params = count_lora_parameters(generator)
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"LoRA percentage: {lora_params / total_params * 100:.2f}%")

    # Check that LoRA was injected
    assert lora_params > 0, "LoRA parameters should be > 0"

    # Test forward pass
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, 192, seq_len)
    g = torch.randn(batch_size, 256, 1)

    output = generator(x, g=g)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape[0] == batch_size
    assert output.shape[1] == 1

    # Check trainable parameters
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in generator.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")

    print("GeneratorLoRA with LoRA tests passed!")
    return True


def test_synthesizer_lora_wrapper():
    """Test SynthesizerLoRA wrapper with a mock synthesizer."""
    print("\n" + "=" * 60)
    print("Testing SynthesizerLoRA Wrapper")
    print("=" * 60)

    # Create a mock synthesizer for testing
    class MockSynthesizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec_channels = 1025
            self.inter_channels = 192
            self.hidden_channels = 192
            self.gin_channels = 256
            self.segment_size = 32

            # Mock decoder (Generator-like)
            self.dec = nn.Sequential(
                nn.Conv1d(192, 256, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 1, 3, padding=1),
            )

            # Mock encoder
            self.enc_p = nn.Linear(256, 192)

            # Mock flow
            self.flow = nn.Linear(192, 192)

            # Mock embedding
            self.emb_g = nn.Embedding(100, 256)

        def forward(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds=None):
            # Simplified forward for testing
            batch_size = phone.shape[0]
            g = self.emb_g(ds).unsqueeze(-1) if ds is not None else None
            z = torch.randn(batch_size, 192, 32)
            o = self.dec(z)
            return o, None, None, None, None

        def infer(self, phone, phone_lengths, pitch, nsff0, sid,
                  skip_head=None, return_length=None, return_length2=None):
            batch_size = phone.shape[0]
            z = torch.randn(batch_size, 192, 100)
            o = self.dec(z)
            return o, None, None

        def remove_weight_norm(self):
            pass

    # Create mock synthesizer
    mock_synth = MockSynthesizer()

    # Create LoRA config
    lora_config = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["dec"],
    )

    # Wrap with SynthesizerLoRA
    synth_lora = SynthesizerLoRA(
        base_synthesizer=mock_synth,
        lora_config=lora_config,
        freeze_non_lora=True,
    )

    # Check attributes were copied
    assert synth_lora.inter_channels == 192
    assert synth_lora.gin_channels == 256

    # Count parameters
    lora_params = synth_lora.get_lora_parameters()
    print(f"Number of LoRA parameter tensors: {len(lora_params)}")

    # Test forward pass
    batch_size = 2
    phone = torch.randn(batch_size, 100, 256)
    phone_lengths = torch.tensor([100, 100])
    pitch = torch.randint(0, 256, (batch_size, 100))
    pitchf = torch.randn(batch_size, 100)
    y = torch.randn(batch_size, 1025, 200)
    y_lengths = torch.tensor([200, 200])
    ds = torch.tensor([0, 1])

    output = synth_lora.forward(phone, phone_lengths, pitch, pitchf, y, y_lengths, ds)
    print(f"Forward output shape: {output[0].shape}")

    # Test infer
    nsff0 = torch.randn(batch_size, 100)
    sid = torch.tensor([0, 1])
    infer_output = synth_lora.infer(phone, phone_lengths, pitch, nsff0, sid)
    print(f"Infer output shape: {infer_output[0].shape}")

    # Test get_lora_state_dict
    lora_state = synth_lora.get_lora_state_dict()
    print(f"LoRA state dict keys: {len(lora_state)}")

    # Test freeze/unfreeze
    synth_lora.unfreeze_all()
    trainable_after_unfreeze = sum(1 for p in synth_lora.synthesizer.parameters() if p.requires_grad)

    synth_lora.freeze_base_model()
    trainable_after_freeze = sum(1 for p in synth_lora.synthesizer.parameters() if p.requires_grad)

    print(f"Trainable params after unfreeze: {trainable_after_unfreeze}")
    print(f"Trainable params after freeze: {trainable_after_freeze}")

    print("SynthesizerLoRA wrapper tests passed!")
    return True


def test_generator_resblock_types():
    """Test GeneratorLoRA with different ResBlock types."""
    print("\n" + "=" * 60)
    print("Testing GeneratorLoRA with ResBlock Types")
    print("=" * 60)

    for resblock_type in ["1", "2"]:
        print(f"\nTesting with ResBlock{resblock_type}...")

        generator = GeneratorLoRA(
            initial_channel=192,
            resblock=resblock_type,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]] if resblock_type == "1" else [[1, 3], [1, 3], [1, 3]],
            upsample_rates=[10, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 20, 4, 4],
            gin_channels=256,
            lora_config=None,
        )

        # Test forward
        x = torch.randn(1, 192, 50)
        g = torch.randn(1, 256, 1)
        output = generator(x, g=g)

        print(f"  ResBlock{resblock_type}: Input {x.shape} -> Output {output.shape}")

    print("\nResBlock type tests passed!")
    return True


def test_lora_parameter_freezing():
    """Test that LoRA correctly freezes base parameters."""
    print("\n" + "=" * 60)
    print("Testing LoRA Parameter Freezing")
    print("=" * 60)

    lora_config = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["ups", "resblocks"],
    )

    generator = GeneratorLoRA(
        initial_channel=192,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 20, 4, 4],
        gin_channels=256,
        lora_config=lora_config,
    )

    # Check parameter status
    lora_trainable = 0
    lora_frozen = 0
    base_trainable = 0
    base_frozen = 0

    for name, param in generator.named_parameters():
        if 'lora_' in name:
            if param.requires_grad:
                lora_trainable += param.numel()
            else:
                lora_frozen += param.numel()
        else:
            if param.requires_grad:
                base_trainable += param.numel()
            else:
                base_frozen += param.numel()

    print(f"LoRA trainable: {lora_trainable:,}")
    print(f"LoRA frozen: {lora_frozen:,}")
    print(f"Base trainable: {base_trainable:,}")
    print(f"Base frozen: {base_frozen:,}")

    # LoRA params should be trainable
    assert lora_trainable > 0, "LoRA parameters should be trainable"
    # LoRA params should not be frozen
    assert lora_frozen == 0, "No LoRA parameters should be frozen"

    print("Parameter freezing tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RVC-LoRA Model Integration Tests")
    print("=" * 60)

    tests = [
        ("ResBlock1", test_resblock1),
        ("ResBlock2", test_resblock2),
        ("GeneratorLoRA Creation", test_generator_lora_creation),
        ("GeneratorLoRA with LoRA", test_generator_lora_with_lora),
        ("SynthesizerLoRA Wrapper", test_synthesizer_lora_wrapper),
        ("Generator ResBlock Types", test_generator_resblock_types),
        ("LoRA Parameter Freezing", test_lora_parameter_freezing),
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
