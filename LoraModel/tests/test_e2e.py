"""
阶段 5: 端到端测试脚本
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加 RVC 根目录
rvc_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, rvc_root)

import torch
import numpy as np

# 40k 模型的默认配置
DEFAULT_CONFIG_40K = {
    "spec_channels": 1025,
    "segment_size": 32,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 10, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "spk_embed_dim": 109,
    "gin_channels": 256,
    "sr": 40000,
}


def test_load_base_model():
    """测试 1: 加载底模"""
    print('[Test 1] 加载底模 G40k.pth...')
    try:
        checkpoint = torch.load(
            'download/pretrained_v2/G40k.pth',
            map_location='cpu',
            weights_only=False
        )
        print(f'  - 检查点类型: {type(checkpoint)}')
        if isinstance(checkpoint, dict):
            print(f'  - 键: {list(checkpoint.keys())}')
        print('  [PASS] 底模加载成功')
        return checkpoint
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return None


def test_create_synthesizer(checkpoint):
    """测试 2: 创建 Synthesizer 模型"""
    print('\n[Test 2] 创建 Synthesizer 模型...')
    try:
        # 底模是 v2 版本 (768 维特征)
        from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid

        config = DEFAULT_CONFIG_40K
        print(f'  - 使用 40k v2 配置 (768 维)')

        model = SynthesizerTrnMs768NSFsid(
            spec_channels=config["spec_channels"],
            segment_size=config["segment_size"],
            inter_channels=config["inter_channels"],
            hidden_channels=config["hidden_channels"],
            filter_channels=config["filter_channels"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            kernel_size=config["kernel_size"],
            p_dropout=config["p_dropout"],
            resblock=config["resblock"],
            resblock_kernel_sizes=config["resblock_kernel_sizes"],
            resblock_dilation_sizes=config["resblock_dilation_sizes"],
            upsample_rates=config["upsample_rates"],
            upsample_initial_channel=config["upsample_initial_channel"],
            upsample_kernel_sizes=config["upsample_kernel_sizes"],
            spk_embed_dim=config["spk_embed_dim"],
            gin_channels=config["gin_channels"],
            sr=config["sr"],
            is_half=False,
        )
        print(f'  - 模型创建成功')

        total_params = sum(p.numel() for p in model.parameters())
        print(f'  - 参数量: {total_params:,}')

        # 加载权重
        state_dict = checkpoint['model']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'  - 缺失键: {len(missing)}')
        print(f'  - 多余键: {len(unexpected)}')

        if missing:
            print(f'  - 缺失键示例: {missing[:3]}')

        print('  [PASS] 模型创建并加载成功')
        return model
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return None


def test_inject_lora(model):
    """测试 3: 注入 LoRA"""
    print('\n[Test 3] 注入 LoRA 层...')
    try:
        from lora import inject_lora, LoRAConfig

        # 先移除 weight_norm
        print('  - 移除 weight_norm...')
        model.remove_weight_norm()

        # 使用默认配置
        config = LoRAConfig(r=8, lora_alpha=16)

        # 注入前统计
        before_params = sum(p.numel() for p in model.parameters())

        # 注入 LoRA 到 decoder (Generator)
        print('  - 注入 LoRA 到 dec (Generator)...')
        inject_lora(model.dec, config, target_modules=['ups', 'resblocks'])

        # 注入后统计
        after_params = sum(p.numel() for p in model.parameters())
        lora_params = after_params - before_params

        # 统计 LoRA 层数
        lora_layer_count = sum(1 for name, _ in model.dec.named_modules() if 'lora_' in name.lower() or hasattr(_, 'lora_A'))

        print(f'  - 注入前参数: {before_params:,}')
        print(f'  - 注入后参数: {after_params:,}')
        print(f'  - LoRA 参数: {lora_params:,}')
        print(f'  - LoRA 占比: {lora_params/after_params*100:.2f}%')

        print('  [PASS] LoRA 注入成功')
        return True
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试 4: 前向传播 (推理模式)"""
    print('\n[Test 4] 测试前向传播 (推理)...')
    try:
        model.eval()

        batch_size = 1
        seq_len = 100

        # 创建输入张量
        phone = torch.randn(batch_size, seq_len, 768)  # phone features (v2: 768 dim)
        phone_lengths = torch.tensor([seq_len])
        pitch = torch.randint(0, 256, (batch_size, seq_len))  # quantized pitch
        nsff0 = torch.randn(batch_size, seq_len) * 100 + 200  # continuous f0
        sid = torch.tensor([0])  # speaker id

        print(f'  - phone: {phone.shape}')
        print(f'  - pitch: {pitch.shape}')
        print(f'  - nsff0: {nsff0.shape}')
        print(f'  - sid: {sid.shape}')

        with torch.no_grad():
            output = model.infer(phone, phone_lengths, pitch, nsff0, sid)

        if isinstance(output, tuple):
            audio = output[0]
        else:
            audio = output

        print(f'  - 输出形状: {audio.shape}')
        print(f'  - 输出范围: [{audio.min().item():.4f}, {audio.max().item():.4f}]')

        print('  [PASS] 前向传播成功')
        return True
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return False


def test_save_load_lora(model):
    """测试 5: 保存和加载 LoRA 权重"""
    print('\n[Test 5] 保存和加载 LoRA 权重...')
    try:
        from lora import extract_lora_weights

        # 提取 LoRA 权重
        lora_state = extract_lora_weights(model)
        print(f'  - 提取的 LoRA 键数: {len(lora_state)}')

        if len(lora_state) == 0:
            # 尝试从 dec 提取
            lora_state = extract_lora_weights(model.dec)
            print(f'  - 从 dec 提取的 LoRA 键数: {len(lora_state)}')

        # 计算大小
        total_size = sum(v.numel() * v.element_size() for v in lora_state.values())
        print(f'  - LoRA 权重大小: {total_size / 1024:.2f} KB')

        # 保存
        save_path = 'download/test_lora_weights.pth'
        torch.save(lora_state, save_path)
        file_size = os.path.getsize(save_path)
        print(f'  - 保存到: {save_path} ({file_size / 1024:.2f} KB)')

        # 重新加载
        loaded_state = torch.load(save_path, map_location='cpu', weights_only=True)
        print(f'  - 重新加载成功, 键数: {len(loaded_state)}')

        # 清理
        os.remove(save_path)

        print('  [PASS] LoRA 保存/加载成功')
        return True
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return False


def test_freeze_unfreeze(model):
    """测试 6: 冻结/解冻参数"""
    print('\n[Test 6] 测试参数冻结/解冻...')
    try:
        # 冻结非 LoRA 参数
        lora_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        total = lora_params + frozen_params

        print(f'  - 总参数: {total:,}')
        print(f'  - 可训练 (LoRA): {lora_params:,}')
        print(f'  - 已冻结: {frozen_params:,}')
        print(f'  - 可训练占比: {lora_params/total*100:.2f}%')

        print('  [PASS] 参数冻结/解冻成功')
        return True
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return False


def test_audio_loading():
    """测试 7: 加载测试音频"""
    print('\n[Test 7] 加载测试音频...')
    try:
        import librosa

        audio_files = [f'download/{i}.wav' for i in range(1, 8)]

        for audio_file in audio_files[:3]:  # 只测试前3个
            if os.path.exists(audio_file):
                audio, sr = librosa.load(audio_file, sr=None)
                print(f'  - {audio_file}: sr={sr}, len={len(audio)}, dur={len(audio)/sr:.2f}s')

        print('  [PASS] 音频加载成功')
        return True
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return False


def test_synthesizer_lora_wrapper(model):
    """测试 8: SynthesizerLoRA 包装器"""
    print('\n[Test 8] 测试 SynthesizerLoRA 包装器...')
    try:
        from models import SynthesizerLoRA
        from lora import LoRAConfig

        # 重新加载一个干净的模型 (v2: 768 维)
        from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid

        config = DEFAULT_CONFIG_40K
        fresh_model = SynthesizerTrnMs768NSFsid(
            spec_channels=config["spec_channels"],
            segment_size=config["segment_size"],
            inter_channels=config["inter_channels"],
            hidden_channels=config["hidden_channels"],
            filter_channels=config["filter_channels"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            kernel_size=config["kernel_size"],
            p_dropout=config["p_dropout"],
            resblock=config["resblock"],
            resblock_kernel_sizes=config["resblock_kernel_sizes"],
            resblock_dilation_sizes=config["resblock_dilation_sizes"],
            upsample_rates=config["upsample_rates"],
            upsample_initial_channel=config["upsample_initial_channel"],
            upsample_kernel_sizes=config["upsample_kernel_sizes"],
            spk_embed_dim=config["spk_embed_dim"],
            gin_channels=config["gin_channels"],
            sr=config["sr"],
            is_half=False,
        )

        # 加载权重
        checkpoint = torch.load('download/pretrained_v2/G40k.pth', map_location='cpu', weights_only=False)
        fresh_model.load_state_dict(checkpoint['model'], strict=False)

        # 创建 LoRA 配置
        lora_config = LoRAConfig(r=8, lora_alpha=16, target_modules=['ups', 'resblocks'])

        # 创建 SynthesizerLoRA 包装器
        synth_lora = SynthesizerLoRA(
            base_synthesizer=fresh_model,
            lora_config=lora_config,
            freeze_non_lora=True
        )

        print(f'  - SynthesizerLoRA 创建成功')

        # 获取 LoRA 参数
        lora_params = synth_lora.get_lora_parameters()
        print(f'  - LoRA 参数组数: {len(lora_params)}')

        # 获取 LoRA state dict
        lora_state = synth_lora.get_lora_state_dict()
        print(f'  - LoRA state dict 键数: {len(lora_state)}')

        print('  [PASS] SynthesizerLoRA 包装器测试成功')
        return True
    except Exception as e:
        print(f'  [FAIL] {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    print('=' * 50)
    print('RVC-LoRA 阶段 5: 端到端测试')
    print('=' * 50)
    print()

    results = []

    # 测试 1: 加载底模
    checkpoint = test_load_base_model()
    results.append(('加载底模', checkpoint is not None))

    if checkpoint is None:
        print('\n底模加载失败，无法继续测试')
        return

    # 测试 2: 创建模型
    model = test_create_synthesizer(checkpoint)
    results.append(('创建 Synthesizer', model is not None))

    if model is None:
        print('\n模型创建失败，无法继续测试')
        return

    # 测试 3: 注入 LoRA
    lora_ok = test_inject_lora(model)
    results.append(('注入 LoRA', lora_ok))

    # 测试 4: 前向传播
    forward_ok = test_forward_pass(model)
    results.append(('前向传播', forward_ok))

    # 测试 5: 保存/加载 LoRA
    save_load_ok = test_save_load_lora(model)
    results.append(('保存/加载 LoRA', save_load_ok))

    # 测试 6: 冻结/解冻
    freeze_ok = test_freeze_unfreeze(model)
    results.append(('冻结/解冻', freeze_ok))

    # 测试 7: 音频加载
    audio_ok = test_audio_loading()
    results.append(('音频加载', audio_ok))

    # 测试 8: SynthesizerLoRA 包装器
    wrapper_ok = test_synthesizer_lora_wrapper(model)
    results.append(('SynthesizerLoRA 包装器', wrapper_ok))

    # 汇总
    print('\n' + '=' * 50)
    print('测试结果汇总')
    print('=' * 50)

    passed = 0
    for name, ok in results:
        status = 'PASS' if ok else 'FAIL'
        print(f'  {name}: [{status}]')
        if ok:
            passed += 1

    print(f'\n总计: {passed}/{len(results)} 通过')
    print('=' * 50)


if __name__ == '__main__':
    main()
