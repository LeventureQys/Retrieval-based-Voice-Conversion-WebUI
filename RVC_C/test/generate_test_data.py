"""
使用 Python 生成的中间数据测试 C++ 合成器
保存所有中间数据为二进制文件，供 C++ 读取
"""

import os
import sys
import numpy as np
import struct

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

import onnxruntime as ort

def save_binary(filepath, data, dtype=np.float32):
    """保存为简单的二进制格式"""
    data = np.asarray(data, dtype=dtype)
    with open(filepath, 'wb') as f:
        # 写入维度数
        f.write(struct.pack('<I', len(data.shape)))
        # 写入每个维度
        for dim in data.shape:
            f.write(struct.pack('<Q', dim))
        # 写入数据
        f.write(data.tobytes())
    print(f"Saved {filepath}: shape={data.shape}, dtype={dtype}")

def main():
    print("="*60)
    print("  Generate Test Data for C++ Comparison")
    print("="*60)

    # 加载 Python 参考数据
    ref_phone = np.load('RVC_C/test/ref_phone.npy')
    ref_pitch = np.load('RVC_C/test/ref_pitch.npy')
    ref_pitchf = np.load('RVC_C/test/ref_pitchf.npy')

    print(f"\nPython reference data:")
    print(f"  phone: {ref_phone.shape}")
    print(f"  pitch: {ref_pitch.shape}")
    print(f"  pitchf: {ref_pitchf.shape}")

    # 使用固定随机种子生成 rnd
    np.random.seed(42)
    synth_frames = ref_phone.shape[1]
    rnd = np.random.randn(1, 192, synth_frames).astype(np.float32)

    print(f"  rnd: {rnd.shape}")

    # 保存为二进制文件
    save_binary('RVC_C/test/bin_phone.bin', ref_phone, np.float32)
    save_binary('RVC_C/test/bin_pitch.bin', ref_pitch, np.int64)
    save_binary('RVC_C/test/bin_pitchf.bin', ref_pitchf, np.float32)
    save_binary('RVC_C/test/bin_rnd.bin', rnd, np.float32)

    # 运行 Python 合成器
    print("\n" + "="*60)
    print("  Running Python Synthesizer")
    print("="*60)

    synthesizer_path = "RVC_C/test/models/Rem_e440_s38720.onnx"
    synth_sess = ort.InferenceSession(synthesizer_path, providers=['CPUExecutionProvider'])

    phone_lengths = np.array([synth_frames], dtype=np.int64)
    ds = np.array([0], dtype=np.int64)

    out_wav = synth_sess.run(None, {
        'phone': ref_phone.astype(np.float32),
        'phone_lengths': phone_lengths,
        'pitch': ref_pitch.astype(np.int64),
        'pitchf': ref_pitchf.astype(np.float32),
        'ds': ds,
        'rnd': rnd
    })[0]

    print(f"\nOutput shape: {out_wav.shape}")
    print(f"Output range: [{out_wav.min():.4f}, {out_wav.max():.4f}]")

    # 保存输出
    save_binary('RVC_C/test/bin_output_python.bin', out_wav.squeeze(), np.float32)

    # 也保存为 WAV
    audio_out = out_wav.squeeze()
    with open('RVC_C/test/python_with_fixed_rnd.wav', 'wb') as f:
        f.write(b'RIFF')
        data_size = len(audio_out) * 4
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 3))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', 48000))
        f.write(struct.pack('<I', 48000 * 4))
        f.write(struct.pack('<H', 4))
        f.write(struct.pack('<H', 32))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio_out.astype(np.float32).tobytes())

    print(f"\nSaved: RVC_C/test/python_with_fixed_rnd.wav")

    print("\n" + "="*60)
    print("  Done! Now run C++ with these inputs.")
    print("="*60)


if __name__ == "__main__":
    main()
