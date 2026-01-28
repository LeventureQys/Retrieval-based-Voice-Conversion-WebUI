"""
Simple example of using LoRA core functionality

This example demonstrates how to use the LoRA implementation
without requiring the full RVC model.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('..')

from lora import (
    LoRAConfig,
    inject_lora,
    save_lora_checkpoint,
    load_lora_checkpoint,
    merge_lora_weights,
    print_lora_info,
    DEFAULT_CONFIG,
    HIGH_QUALITY_CONFIG,
    FAST_CONFIG,
)


def example_1_basic_usage():
    """Example 1: Basic LoRA usage with a simple model."""
    print("\n" + "="*60)
    print("Example 1: Basic LoRA Usage")
    print("="*60)

    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create model
    model = SimpleModel()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure LoRA
    config = LoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["fc"]  # Target all linear layers with "fc" in name
    )

    # Inject LoRA
    model = inject_lora(model, config)

    # Print LoRA info
    print_lora_info(model)

    # Test forward pass
    x = torch.randn(4, 128)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def example_2_training():
    """Example 2: Training with LoRA."""
    print("\n" + "="*60)
    print("Example 2: Training with LoRA")
    print("="*60)

    # Create model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # Inject LoRA
    config = LoRAConfig(r=4, lora_alpha=8, target_modules=["fc"])
    model = inject_lora(model, config)

    # Setup optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    # Simple training loop
    print("Training for 10 steps...")
    for step in range(10):
        # Generate dummy data
        x = torch.randn(4, 10)
        target = torch.randn(4, 10)

        # Forward pass
        output = model(x)
        loss = nn.functional.mse_loss(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("Training complete!")


def example_3_save_load():
    """Example 3: Saving and loading LoRA checkpoints."""
    print("\n" + "="*60)
    print("Example 3: Save and Load LoRA Checkpoints")
    print("="*60)

    # Create and train model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    config = LoRAConfig(r=4, lora_alpha=8, target_modules=["fc"])
    model = inject_lora(model, config)

    # Save checkpoint
    checkpoint_path = "example_lora.pth"
    save_lora_checkpoint(
        model=model,
        path=checkpoint_path,
        config=config,
        epoch=100,
        additional_info={"note": "Example checkpoint"}
    )
    print(f"Checkpoint saved to {checkpoint_path}")

    # Create new model and load checkpoint
    model2 = SimpleModel()
    model2 = inject_lora(model2, config)

    model2, _, epoch, loaded_config = load_lora_checkpoint(
        model=model2,
        path=checkpoint_path
    )
    print(f"Checkpoint loaded, epoch: {epoch}")

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Cleaned up {checkpoint_path}")


def example_4_weight_merging():
    """Example 4: Weight merging for inference."""
    print("\n" + "="*60)
    print("Example 4: Weight Merging for Inference")
    print("="*60)

    # Create model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    config = LoRAConfig(r=4, lora_alpha=8, target_modules=["fc"], merge_weights=True)
    model = inject_lora(model, config)

    # Test input
    x = torch.randn(4, 10)

    # Training mode (unmerged)
    model.train()
    output_train = model(x)
    print("Training mode (LoRA separate):")
    print(f"  Output mean: {output_train.mean().item():.4f}")

    # Inference mode (merged)
    model.eval()
    output_eval = model(x)
    print("Inference mode (LoRA merged):")
    print(f"  Output mean: {output_eval.mean().item():.4f}")

    # Outputs should be the same
    print(f"Outputs match: {torch.allclose(output_train, output_eval, atol=1e-6)}")


def example_5_predefined_configs():
    """Example 5: Using predefined configurations."""
    print("\n" + "="*60)
    print("Example 5: Predefined Configurations")
    print("="*60)

    configs = {
        "DEFAULT": DEFAULT_CONFIG,
        "HIGH_QUALITY": HIGH_QUALITY_CONFIG,
        "FAST": FAST_CONFIG,
    }

    for name, config in configs.items():
        print(f"\n{name} Config:")
        print(f"  Rank: {config.r}")
        print(f"  Alpha: {config.lora_alpha}")
        print(f"  Ups rank: {config.ups_rank}")
        print(f"  ResBlock rank: {config.resblock_rank}")
        print(f"  Target modules: {config.target_modules}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RVC-LoRA Usage Examples")
    print("="*60)

    examples = [
        example_1_basic_usage,
        example_2_training,
        example_3_save_load,
        example_4_weight_merging,
        example_5_predefined_configs,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
