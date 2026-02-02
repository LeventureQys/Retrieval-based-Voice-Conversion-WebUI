"""
RVC C++ 实现端到端测试脚本
对比 Python 和 C++ 的合成器输出
"""

import os
import sys
import numpy as np
import struct

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def load_wav(filepath):
    """简单的 WAV 文件加载"""
    with open(filepath, 'rb') as f:
        # 读取 RIFF 头
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError("Not a valid WAV file")

        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError("Not a valid WAV file")

        # 查找 fmt 和 data 块
        sample_rate = 16000
        channels = 1
        bits_per_sample = 16
        audio_data = None

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                audio_format = struct.unpack('<H', f.read(2))[0]
                channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                byte_rate = struct.unpack('<I', f.read(4))[0]
                block_align = struct.unpack('<H', f.read(2))[0]
                bits_per_sample = struct.unpack('<H', f.read(2))[0]
                # 跳过额外的 fmt 数据
                if chunk_size > 16:
                    f.read(chunk_size - 16)
            elif chunk_id == b'data':
                if bits_per_sample == 16:
                    num_samples = chunk_size // 2
                    audio_data = np.frombuffer(f.read(chunk_size), dtype=np.int16).astype(np.float32) / 32768.0
                elif bits_per_sample == 32:
                    num_samples = chunk_size // 4
                    audio_data = np.frombuffer(f.read(chunk_size), dtype=np.float32)
                else:
                    f.read(chunk_size)
            else:
                f.read(chunk_size)

        if audio_data is None:
            raise ValueError("No audio data found")

        # 转换为单声道
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

        return audio_data, sample_rate


def save_wav(filepath, audio, sample_rate=48000):
    """保存 WAV 文件"""
    with open(filepath, 'wb') as f:
        # RIFF 头
        f.write(b'RIFF')
        data_size = len(audio) * 4
        file_size = 36 + data_size
        f.write(struct.pack('<I', file_size))
        f.write(b'WAVE')

        # fmt 块
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 3))   # audio format (IEEE float)
        f.write(struct.pack('<H', 1))   # channels
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))  # byte rate
        f.write(struct.pack('<H', 4))   # block align
        f.write(struct.pack('<H', 32))  # bits per sample

        # data 块
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio.astype(np.float32).tobytes())


def test_synthesizer_python(model_path, time_steps=100, hidden_channels=768):
    """使用 Python/ONNX Runtime 测试合成器"""
    import onnxruntime as ort

    print(f"\n=== Python ONNX Runtime Test ===")
    print(f"Loading model: {model_path}")

    # 加载模型
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # 打印输入信息
    print("Model inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape}, {inp.type}")

    # 准备测试数据
    np.random.seed(42)  # 固定随机种子以便对比

    phone = np.random.randn(1, time_steps, hidden_channels).astype(np.float32) * 0.1
    phone_lengths = np.array([time_steps], dtype=np.int64)

    # 生成音高数据
    pitch = np.zeros((1, time_steps), dtype=np.int64)
    pitchf = np.zeros((1, time_steps), dtype=np.float32)
    for t in range(time_steps):
        freq = 220.0 + 10.0 * np.sin(t * 0.05)
        pitch_val = int(12.0 * np.log2(freq / 10.0))
        pitch_val = max(1, min(255, pitch_val))
        pitch[0, t] = pitch_val
        pitchf[0, t] = freq

    ds = np.array([0], dtype=np.int64)
    rnd = (np.random.randn(1, 192, time_steps) * 0.1).astype(np.float32)

    print(f"Input shapes:")
    print(f"  phone: {phone.shape}")
    print(f"  phone_lengths: {phone_lengths.shape}")
    print(f"  pitch: {pitch.shape}")
    print(f"  pitchf: {pitchf.shape}")
    print(f"  ds: {ds.shape}")
    print(f"  rnd: {rnd.shape}")

    # 运行推理
    import time
    start = time.time()

    outputs = sess.run(None, {
        'phone': phone,
        'phone_lengths': phone_lengths,
        'pitch': pitch,
        'pitchf': pitchf,
        'ds': ds,
        'rnd': rnd
    })

    elapsed = (time.time() - start) * 1000

    output = outputs[0]
    print(f"\nInference completed in {elapsed:.1f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output mean: {output.mean():.4f}")

    return output


def test_f0_extraction(audio_path):
    """测试 F0 提取"""
    print(f"\n=== F0 Extraction Test ===")

    # 加载音频
    audio, sr = load_wav(audio_path)
    print(f"Loaded audio: {len(audio)} samples, {sr} Hz")

    # 使用 pyworld 提取 F0
    try:
        import pyworld as pw

        # 转换为 double
        audio_double = audio.astype(np.float64)

        # 提取 F0
        import time
        start = time.time()
        f0, t = pw.harvest(audio_double, sr, f0_floor=71.0, f0_ceil=800.0, frame_period=5.0)
        elapsed = (time.time() - start) * 1000

        print(f"F0 extraction completed in {elapsed:.1f} ms")
        print(f"F0 frames: {len(f0)}")

        # 统计
        voiced = f0 > 0
        if voiced.sum() > 0:
            print(f"Voiced frames: {voiced.sum()}")
            print(f"Average F0: {f0[voiced].mean():.1f} Hz")
            print(f"F0 range: [{f0[voiced].min():.1f}, {f0[voiced].max():.1f}] Hz")

        return f0

    except ImportError:
        print("pyworld not installed, skipping F0 test")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RVC C++ End-to-End Test")
    parser.add_argument("--model", "-m",
                        default="RVC_C/test/models/Rem_e440_s38720.onnx",
                        help="Path to synthesizer ONNX model")
    parser.add_argument("--audio", "-a",
                        default="RVC_C/test/test_voice/7.wav",
                        help="Path to test audio file")
    parser.add_argument("--output", "-o",
                        default="RVC_C/test/python_output.wav",
                        help="Path to output audio file")

    args = parser.parse_args()

    os.chdir(project_root)

    print("="*50)
    print("  RVC C++ Implementation Test")
    print("="*50)

    # 测试合成器
    if os.path.exists(args.model):
        output = test_synthesizer_python(args.model)

        # 保存输出
        if output is not None:
            # 输出形状是 [1, 1, samples]，需要 squeeze
            audio_out = output.squeeze()
            save_wav(args.output, audio_out, sample_rate=48000)
            print(f"\nSaved output to: {args.output}")
    else:
        print(f"Model not found: {args.model}")

    # 测试 F0 提取
    if os.path.exists(args.audio):
        test_f0_extraction(args.audio)
    else:
        print(f"Audio file not found: {args.audio}")

    print("\n" + "="*50)
    print("  Test completed!")
    print("="*50)


if __name__ == "__main__":
    main()
