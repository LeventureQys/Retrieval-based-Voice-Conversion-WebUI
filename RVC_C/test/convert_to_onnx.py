"""
RVC PyTorch 模型转 ONNX 格式脚本
用于将 .pth 模型转换为 C++ 推理可用的 ONNX 格式
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


def convert_pth_to_onnx(pth_path, onnx_path, device="cpu"):
    """
    将 RVC .pth 模型转换为 ONNX 格式

    Args:
        pth_path: 输入的 .pth 模型路径
        onnx_path: 输出的 .onnx 模型路径
        device: 导出设备
    """
    print(f"Loading model from: {pth_path}")

    # 加载模型
    cpt = torch.load(pth_path, map_location="cpu")

    # 获取配置
    config = cpt["config"]

    # 修正说话人数量
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]

    print(f"Model config: {config}")

    # 确定 hidden_channels (v1=256, v2=768)
    # 从权重中推断
    if "enc_p.emb_phone.weight" in cpt["weight"]:
        hidden_channels = cpt["weight"]["enc_p.emb_phone.weight"].shape[1]
    else:
        # 默认使用 256
        hidden_channels = 256

    print(f"Hidden channels: {hidden_channels}")

    # 确定版本 (v1=256, v2=768)
    version = "v2" if hidden_channels == 768 else "v1"
    print(f"Model version: {version}")

    # 创建模型 - 添加 version 参数
    net_g = SynthesizerTrnMsNSFsidM(*config, version=version, is_half=False)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()
    net_g.to(device)

    # 创建测试输入
    test_length = 200
    test_phone = torch.rand(1, test_length, hidden_channels).to(device)
    test_phone_lengths = torch.tensor([test_length]).long().to(device)
    test_pitch = torch.randint(size=(1, test_length), low=5, high=255).to(device)
    test_pitchf = torch.rand(1, test_length).to(device)
    test_ds = torch.LongTensor([0]).to(device)
    test_rnd = torch.rand(1, 192, test_length).to(device)

    # 导出 ONNX
    print(f"Exporting to: {onnx_path}")

    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]

    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        onnx_path,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    print(f"Export completed: {onnx_path}")

    # 验证导出的模型
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model validation passed!")
    except ImportError:
        print("onnx package not installed, skipping validation")
    except Exception as e:
        print(f"ONNX validation warning: {e}")

    return True


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")

    # 查找所有 .pth 文件
    pth_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]

    if not pth_files:
        print("No .pth files found in models directory")
        return

    print(f"Found {len(pth_files)} .pth file(s)")

    for pth_file in pth_files:
        pth_path = os.path.join(models_dir, pth_file)
        onnx_file = pth_file.replace(".pth", ".onnx")
        onnx_path = os.path.join(models_dir, onnx_file)

        try:
            convert_pth_to_onnx(pth_path, onnx_path)
        except Exception as e:
            print(f"Error converting {pth_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
