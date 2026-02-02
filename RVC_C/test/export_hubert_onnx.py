"""
HuBERT 模型导出为 ONNX 格式
用于 RVC C++ 推理
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class HuBERTONNXWrapper(nn.Module):
    """
    HuBERT 模型的 ONNX 导出包装器
    简化了原始模型，只保留推理所需的部分
    """
    def __init__(self, hubert_model, output_layer=12, use_final_proj=False):
        super().__init__()
        self.hubert = hubert_model
        self.output_layer = output_layer
        self.use_final_proj = use_final_proj

    def forward(self, source):
        """
        Args:
            source: [batch, audio_length] 音频波形，16kHz
        Returns:
            features: [batch, time_frames, feature_dim] HuBERT 特征
        """
        # 创建 padding mask (全 False，表示没有 padding)
        padding_mask = torch.zeros(source.shape, dtype=torch.bool, device=source.device)

        # 提取特征
        logits = self.hubert.extract_features(
            source=source,
            padding_mask=padding_mask,
            output_layer=self.output_layer
        )

        feats = logits[0]

        # v1 模型需要 final_proj
        if self.use_final_proj and hasattr(self.hubert, 'final_proj'):
            feats = self.hubert.final_proj(feats)

        return feats


def export_hubert_onnx(
    hubert_path="download/hubert_base.pt",
    output_path="RVC_C/test/models/hubert_base.onnx",
    version="v2",
    device="cpu"
):
    """
    导出 HuBERT 模型为 ONNX 格式

    Args:
        hubert_path: HuBERT 模型路径
        output_path: 输出 ONNX 路径
        version: "v1" 或 "v2"
        device: 设备
    """
    print(f"Loading HuBERT model from: {hubert_path}")

    # 尝试使用 fairseq 加载
    try:
        from fairseq import checkpoint_utils

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path],
            suffix=""
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(device)
        hubert_model.eval()

        print("HuBERT model loaded successfully using fairseq")

    except ImportError:
        print("fairseq not installed, trying alternative loading method...")

        # 尝试直接加载
        checkpoint = torch.load(hubert_path, map_location=device)

        if "model" in checkpoint:
            # 这是一个 fairseq checkpoint
            print("Detected fairseq checkpoint format")
            print("Please install fairseq: pip install fairseq")
            return False
        else:
            print("Unknown checkpoint format")
            return False

    # 确定输出层和是否使用 final_proj
    if version == "v1":
        output_layer = 9
        use_final_proj = True
        feature_dim = 256
    else:
        output_layer = 12
        use_final_proj = False
        feature_dim = 768

    print(f"Version: {version}, Output layer: {output_layer}, Feature dim: {feature_dim}")

    # 创建包装器
    wrapper = HuBERTONNXWrapper(hubert_model, output_layer, use_final_proj)
    wrapper.eval()

    # 创建测试输入 (1秒音频 @ 16kHz)
    test_audio = torch.randn(1, 16000).to(device)

    # 测试前向传播
    print("Testing forward pass...")
    with torch.no_grad():
        test_output = wrapper(test_audio)
        print(f"Output shape: {test_output.shape}")
        print(f"Expected: [1, ~50, {feature_dim}]")  # 16000 / 320 = 50 frames

    # 导出 ONNX
    print(f"Exporting to: {output_path}")

    try:
        torch.onnx.export(
            wrapper,
            test_audio,
            output_path,
            input_names=["source"],
            output_names=["features"],
            dynamic_axes={
                "source": {0: "batch", 1: "audio_length"},
                "features": {0: "batch", 1: "time_frames"}
            },
            opset_version=14,  # 使用较低版本以提高兼容性
            do_constant_folding=True,
            verbose=False
        )
        print("ONNX export completed!")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nTrying alternative export method...")

        # 尝试使用 torch.jit.trace
        try:
            traced = torch.jit.trace(wrapper, test_audio)
            torch.onnx.export(
                traced,
                test_audio,
                output_path,
                input_names=["source"],
                output_names=["features"],
                dynamic_axes={
                    "source": {1: "audio_length"},
                    "features": {1: "time_frames"}
                },
                opset_version=14,
                do_constant_folding=True,
                verbose=False
            )
            print("ONNX export completed using traced model!")
        except Exception as e2:
            print(f"Alternative export also failed: {e2}")
            return False

    # 验证导出的模型
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("ONNX model validation passed!")

        # 打印模型信息
        print(f"\nModel inputs:")
        for inp in model.graph.input:
            print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
        print(f"Model outputs:")
        for out in model.graph.output:
            print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    except ImportError:
        print("onnx package not installed, skipping validation")
    except Exception as e:
        print(f"ONNX validation warning: {e}")

    # 测试 ONNX 推理
    try:
        import onnxruntime as ort

        print("\nTesting ONNX inference...")
        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

        test_input = test_audio.numpy()
        onnx_output = sess.run(None, {"source": test_input})[0]

        print(f"ONNX output shape: {onnx_output.shape}")
        print(f"ONNX output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")

        # 比较 PyTorch 和 ONNX 输出
        with torch.no_grad():
            torch_output = wrapper(test_audio).numpy()

        diff = abs(torch_output - onnx_output).max()
        print(f"Max difference between PyTorch and ONNX: {diff:.6f}")

        if diff < 1e-4:
            print("ONNX inference test PASSED!")
        else:
            print("ONNX inference test WARNING: outputs differ significantly")

    except ImportError:
        print("onnxruntime not installed, skipping inference test")
    except Exception as e:
        print(f"ONNX inference test failed: {e}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export HuBERT to ONNX")
    parser.add_argument("--input", "-i", default="download/hubert_base.pt",
                        help="Input HuBERT model path")
    parser.add_argument("--output", "-o", default="RVC_C/test/models/hubert_base.onnx",
                        help="Output ONNX model path")
    parser.add_argument("--version", "-v", default="v2", choices=["v1", "v2"],
                        help="Model version (v1=256dim, v2=768dim)")

    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(project_root)

    success = export_hubert_onnx(
        hubert_path=args.input,
        output_path=args.output,
        version=args.version
    )

    if success:
        print("\n" + "="*50)
        print("HuBERT ONNX export completed successfully!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("HuBERT ONNX export failed!")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
