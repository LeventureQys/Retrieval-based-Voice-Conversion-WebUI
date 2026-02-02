"""
RVC C++ vs Python 逐层对比调试脚本
用于定位导致声音嘶哑的问题
"""

import os
import sys
import numpy as np
import struct

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

import onnxruntime as ort


def load_wav_simple(filepath):
    """简单的 WAV 文件加载"""
    with open(filepath, 'rb') as f:
        riff = f.read(4)
        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)

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
                if chunk_size > 16:
                    f.read(chunk_size - 16)
            elif chunk_id == b'data':
                if bits_per_sample == 16:
                    audio_data = np.frombuffer(f.read(chunk_size), dtype=np.int16).astype(np.float32) / 32768.0
                elif bits_per_sample == 32:
                    audio_data = np.frombuffer(f.read(chunk_size), dtype=np.float32)
                else:
                    f.read(chunk_size)
            else:
                f.read(chunk_size)

        if channels == 2 and audio_data is not None:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

        return audio_data, sample_rate


def save_wav_simple(filepath, audio, sample_rate=48000):
    """保存 WAV 文件"""
    audio = np.clip(audio, -1.0, 1.0)
    with open(filepath, 'wb') as f:
        f.write(b'RIFF')
        data_size = len(audio) * 4
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 3))  # float
        f.write(struct.pack('<H', 1))  # mono
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))
        f.write(struct.pack('<H', 4))
        f.write(struct.pack('<H', 32))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio.astype(np.float32).tobytes())


def resample_linear(audio, src_rate, dst_rate):
    """简单线性插值重采样"""
    ratio = dst_rate / src_rate
    output_size = int(len(audio) * ratio + 0.5)
    output = np.zeros(output_size, dtype=np.float32)

    for i in range(output_size):
        src_pos = i / ratio
        idx = int(src_pos)
        frac = src_pos - idx

        if idx + 1 < len(audio):
            output[i] = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac
        elif idx < len(audio):
            output[i] = audio[idx]

    return output


def extract_f0_harvest(audio, sr, frame_period=5.0):
    """使用 pyworld 提取 F0"""
    try:
        import pyworld as pw
        audio_double = audio.astype(np.float64)
        f0, t = pw.harvest(audio_double, sr, f0_floor=71.0, f0_ceil=800.0, frame_period=frame_period)
        return f0, t
    except ImportError:
        print("pyworld not installed!")
        return None, None


def compare_arrays(name, arr1, arr2, tolerance=1e-5):
    """比较两个数组"""
    if arr1.shape != arr2.shape:
        print(f"  [WARN] {name}: Shape mismatch! {arr1.shape} vs {arr2.shape}")
        # 尝试截断到相同长度
        min_len = min(len(arr1.flatten()), len(arr2.flatten()))
        arr1 = arr1.flatten()[:min_len]
        arr2 = arr2.flatten()[:min_len]

    diff = np.abs(arr1 - arr2)
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff < tolerance:
        print(f"  [OK] {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        return True
    else:
        print(f"  [DIFF] {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        # 找出差异最大的位置
        max_idx = np.argmax(diff)
        print(f"         Max diff at index {max_idx}: {arr1.flatten()[max_idx]:.6f} vs {arr2.flatten()[max_idx]:.6f}")
        return False


def main():
    print("="*60)
    print("  RVC C++ vs Python Layer-by-Layer Comparison")
    print("="*60)

    # 路径
    contentvec_path = "RVC_C/test/models/vec-768-layer-12.onnx"
    synthesizer_path = "RVC_C/test/models/Rem_e440_s38720.onnx"
    input_wav = "RVC_C/test/test_voice/7.wav"

    # 加载模型
    print("\n[1] Loading models...")
    cv_sess = ort.InferenceSession(contentvec_path, providers=['CPUExecutionProvider'])
    synth_sess = ort.InferenceSession(synthesizer_path, providers=['CPUExecutionProvider'])
    print("    Models loaded.")

    # 加载音频
    print("\n[2] Loading audio...")
    audio, sr = load_wav_simple(input_wav)
    print(f"    Loaded: {len(audio)} samples, {sr} Hz")

    # 重采样到 16kHz
    print("\n[3] Resampling to 16kHz...")
    if sr != 16000:
        audio_16k = resample_linear(audio, sr, 16000)
        print(f"    Resampled: {len(audio_16k)} samples")
    else:
        audio_16k = audio

    # 保存重采样后的音频供 C++ 对比
    save_wav_simple("RVC_C/test/debug_audio_16k.wav", audio_16k, 16000)
    print(f"    Saved: RVC_C/test/debug_audio_16k.wav")

    # 提取 F0
    print("\n[4] Extracting F0...")
    f0, t = extract_f0_harvest(audio, sr)
    if f0 is not None:
        print(f"    F0 frames: {len(f0)}")
        voiced = f0 > 0
        print(f"    Voiced frames: {voiced.sum()}")
        if voiced.sum() > 0:
            print(f"    F0 range: [{f0[voiced].min():.1f}, {f0[voiced].max():.1f}] Hz")
            print(f"    F0 mean: {f0[voiced].mean():.1f} Hz")

        # 保存 F0 供对比
        np.save("RVC_C/test/debug_f0.npy", f0)
        print(f"    Saved: RVC_C/test/debug_f0.npy")

    # ContentVec 推理
    print("\n[5] ContentVec inference...")
    cv_input = audio_16k.reshape(1, 1, -1).astype(np.float32)
    print(f"    Input shape: {cv_input.shape}")
    print(f"    Input range: [{cv_input.min():.4f}, {cv_input.max():.4f}]")

    cv_output = cv_sess.run(None, {cv_sess.get_inputs()[0].name: cv_input})[0]
    print(f"    Output shape: {cv_output.shape}")
    print(f"    Output range: [{cv_output.min():.4f}, {cv_output.max():.4f}]")
    print(f"    Output mean: {cv_output.mean():.4f}")
    print(f"    Output std: {cv_output.std():.4f}")

    # 保存 ContentVec 输出
    np.save("RVC_C/test/debug_contentvec.npy", cv_output)
    print(f"    Saved: RVC_C/test/debug_contentvec.npy")

    # 准备合成器输入
    print("\n[6] Preparing synthesizer inputs...")

    cv_frames = cv_output.shape[1]
    cv_dim = cv_output.shape[2]

    # RVC 的处理: ContentVec 特征需要重复 2 倍
    # 参考 Python 代码: hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1)
    # 原始输出是 [1, frames, 768]，需要变成 [1, frames*2, 768]

    # 方法1: 简单重复每帧
    phone = np.repeat(cv_output, 2, axis=1)  # [1, frames*2, 768]
    synth_frames = phone.shape[1]

    print(f"    Phone shape after repeat: {phone.shape}")

    # 调整 F0 长度以匹配
    if f0 is not None:
        if len(f0) > synth_frames:
            f0_adjusted = f0[:synth_frames]
        else:
            f0_adjusted = np.pad(f0, (0, synth_frames - len(f0)), mode='edge')
        print(f"    F0 adjusted length: {len(f0_adjusted)}")
    else:
        # 生成假的 F0
        f0_adjusted = np.ones(synth_frames) * 200.0

    # 转换 F0 到 pitch 格式
    f0_min = 50.0
    f0_max = 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)

    pitchf = f0_adjusted.astype(np.float32)

    # 计算量化 pitch
    pitch = np.zeros(synth_frames, dtype=np.int64)
    for i in range(synth_frames):
        if pitchf[i] > 0:
            f0_mel = 1127.0 * np.log(1 + pitchf[i] / 700.0)
            f0_mel = (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
            f0_mel = np.clip(f0_mel, 1, 255)
            pitch[i] = int(round(f0_mel))
        else:
            pitch[i] = 0

    print(f"    Pitch range: [{pitch.min()}, {pitch.max()}]")
    print(f"    Pitchf range: [{pitchf.min():.1f}, {pitchf.max():.1f}]")

    # 检查 pitch 中 0 的比例
    zero_ratio = (pitch == 0).sum() / len(pitch)
    print(f"    Pitch zero ratio: {zero_ratio*100:.1f}%")

    # 随机噪声
    np.random.seed(42)
    rnd = np.random.randn(1, 192, synth_frames).astype(np.float32) * 0.1

    # 准备输入
    phone_input = phone.astype(np.float32)
    phone_lengths = np.array([synth_frames], dtype=np.int64)
    pitch_input = pitch.reshape(1, -1)
    pitchf_input = pitchf.reshape(1, -1)
    ds = np.array([0], dtype=np.int64)

    print(f"\n    Synthesizer inputs:")
    print(f"      phone: {phone_input.shape}, range=[{phone_input.min():.4f}, {phone_input.max():.4f}]")
    print(f"      phone_lengths: {phone_lengths}")
    print(f"      pitch: {pitch_input.shape}, range=[{pitch_input.min()}, {pitch_input.max()}]")
    print(f"      pitchf: {pitchf_input.shape}, range=[{pitchf_input.min():.1f}, {pitchf_input.max():.1f}]")
    print(f"      ds: {ds}")
    print(f"      rnd: {rnd.shape}, range=[{rnd.min():.4f}, {rnd.max():.4f}]")

    # 保存输入供 C++ 对比
    np.save("RVC_C/test/debug_phone.npy", phone_input)
    np.save("RVC_C/test/debug_pitch.npy", pitch_input)
    np.save("RVC_C/test/debug_pitchf.npy", pitchf_input)
    np.save("RVC_C/test/debug_rnd.npy", rnd)
    print(f"    Saved debug inputs to RVC_C/test/")

    # 合成器推理
    print("\n[7] Synthesizer inference...")

    synth_output = synth_sess.run(None, {
        'phone': phone_input,
        'phone_lengths': phone_lengths,
        'pitch': pitch_input,
        'pitchf': pitchf_input,
        'ds': ds,
        'rnd': rnd
    })[0]

    print(f"    Output shape: {synth_output.shape}")
    print(f"    Output range: [{synth_output.min():.4f}, {synth_output.max():.4f}]")
    print(f"    Output mean: {synth_output.mean():.6f}")
    print(f"    Output std: {synth_output.std():.4f}")

    # 检查输出中的异常值
    audio_out = synth_output.squeeze()

    # 检查是否有 NaN 或 Inf
    if np.isnan(audio_out).any():
        print(f"    [WARN] Output contains NaN!")
    if np.isinf(audio_out).any():
        print(f"    [WARN] Output contains Inf!")

    # 检查削波
    clip_ratio = ((np.abs(audio_out) > 0.99).sum()) / len(audio_out)
    print(f"    Clipping ratio: {clip_ratio*100:.2f}%")

    # 保存输出
    save_wav_simple("RVC_C/test/debug_python_output.wav", audio_out, 48000)
    np.save("RVC_C/test/debug_synth_output.npy", synth_output)
    print(f"    Saved: RVC_C/test/debug_python_output.wav")

    # 分析频谱
    print("\n[8] Spectral analysis...")

    # 简单的能量分析
    frame_size = 2048
    hop_size = 512
    num_frames = (len(audio_out) - frame_size) // hop_size

    energies = []
    for i in range(num_frames):
        start = i * hop_size
        frame = audio_out[start:start+frame_size]
        energy = np.sqrt(np.mean(frame**2))
        energies.append(energy)

    energies = np.array(energies)
    print(f"    Frame energies: min={energies.min():.4f}, max={energies.max():.4f}, mean={energies.mean():.4f}")

    # 检查能量突变（可能导致嘶哑）
    energy_diff = np.abs(np.diff(energies))
    large_jumps = (energy_diff > 0.1).sum()
    print(f"    Large energy jumps (>0.1): {large_jumps}")

    print("\n" + "="*60)
    print("  Debug files saved. Compare with C++ outputs.")
    print("="*60)

    return {
        'audio_16k': audio_16k,
        'f0': f0,
        'cv_output': cv_output,
        'phone': phone_input,
        'pitch': pitch_input,
        'pitchf': pitchf_input,
        'synth_output': synth_output
    }


if __name__ == "__main__":
    results = main()
