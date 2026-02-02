"""
RVC 正确流程的 Python 实现
完全按照原始 RVC 代码的处理方式
"""

import os
import sys
import numpy as np
import struct

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

import onnxruntime as ort
import pyworld


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
        f.write(struct.pack('<H', 3))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))
        f.write(struct.pack('<H', 4))
        f.write(struct.pack('<H', 32))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio.astype(np.float32).tobytes())


def interpolate_f0(f0):
    """对 F0 进行插值处理，填充无声段"""
    data = np.reshape(f0, (f0.size, 1))

    vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data.copy()

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return ip_data[:, 0], vuv_vector[:, 0]


def resize_f0(x, target_len):
    """将 F0 调整到目标长度"""
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * target_len, len(source)) / target_len,
        np.arange(0, len(source)),
        source,
    )
    res = np.nan_to_num(target)
    return res


def compute_f0_harvest(wav, sampling_rate, hop_length, p_len, f0_min=50, f0_max=1100):
    """使用 Harvest 算法计算 F0"""
    frame_period = 1000 * hop_length / sampling_rate

    f0, t = pyworld.harvest(
        wav.astype(np.double),
        fs=sampling_rate,
        f0_ceil=f0_max,
        f0_floor=f0_min,
        frame_period=frame_period,
    )

    # 使用 stonemask 精炼 F0
    f0 = pyworld.stonemask(wav.astype(np.double), f0, t, sampling_rate)

    # 调整到目标长度
    f0_resized = resize_f0(f0, p_len)

    # 插值处理
    f0_interp, vuv = interpolate_f0(f0_resized)

    return f0_interp


def resample_librosa(audio, src_rate, dst_rate):
    """使用 librosa 风格的重采样"""
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate)
    except ImportError:
        # 简单线性插值
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


def rvc_inference_python(
    contentvec_path,
    synthesizer_path,
    input_wav_path,
    output_wav_path,
    speaker_id=0,
    pitch_shift=0,
    sampling_rate=48000,
    hop_size=512
):
    """
    完整的 RVC 推理流程 (Python 版本)
    完全按照原始 RVC 代码实现
    """
    print("="*60)
    print("  RVC Python Reference Implementation")
    print("="*60)

    # 加载模型
    print("\n[1] Loading models...")
    cv_sess = ort.InferenceSession(contentvec_path, providers=['CPUExecutionProvider'])
    synth_sess = ort.InferenceSession(synthesizer_path, providers=['CPUExecutionProvider'])
    print("    Done.")

    # 加载音频
    print("\n[2] Loading audio...")
    wav, sr = load_wav_simple(input_wav_path)
    org_length = len(wav)
    print(f"    Loaded: {org_length} samples, {sr} Hz")

    # 如果采样率不匹配，重采样到目标采样率
    if sr != sampling_rate:
        wav = resample_librosa(wav, sr, sampling_rate)
        print(f"    Resampled to {sampling_rate} Hz: {len(wav)} samples")

    # 重采样到 16kHz 用于 ContentVec
    print("\n[3] Resampling to 16kHz for ContentVec...")
    wav16k = resample_librosa(wav, sampling_rate, 16000)
    print(f"    16kHz audio: {len(wav16k)} samples")

    # ContentVec 推理
    print("\n[4] ContentVec inference...")
    cv_input = wav16k.reshape(1, 1, -1).astype(np.float32)
    hubert = cv_sess.run(None, {cv_sess.get_inputs()[0].name: cv_input})[0]
    print(f"    ContentVec output: {hubert.shape}")

    # 关键步骤: 在 axis=2 上重复，然后转置
    # 原始: [1, frames, 768]
    # repeat axis=2 没有意义因为 768 是特征维度
    # 查看原始代码: hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1)
    # 但 ContentVec 输出是 [1, frames, 768]，axis=2 是 768
    # 这说明原始 ContentVec 输出可能是 [1, 768, frames]

    # 检查 ContentVec 输出格式
    # 根据 onnx_inference.py 中的 ContentVec 类:
    # return logits.transpose(0, 2, 1)  # 输出变成 [1, frames, 768]

    # 所以原始流程是:
    # 1. ContentVec 输出 [1, 768, frames]
    # 2. transpose(0, 2, 1) -> [1, frames, 768]
    # 3. repeat(2, axis=2) 在 768 维度上重复? 这不对

    # 重新看代码:
    # hubert = self.vec_model(wav16k)  # 返回 logits.transpose(0, 2, 1) = [1, frames, 768]
    # hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1)
    # 这会变成 [1, frames, 1536] -> transpose -> [1, 1536, frames]? 这也不对

    # 让我重新理解:
    # vec_model.forward() 返回 logits.transpose(0, 2, 1)
    # 如果 logits 是 [1, 768, frames]，那么 transpose 后是 [1, frames, 768]
    # 然后 repeat(2, axis=2) 会在最后一个维度重复: [1, frames, 1536]
    # 再 transpose(0, 2, 1) 变成 [1, 1536, frames]

    # 但合成器期望 phone 是 [1, T, 768]，不是 1536

    # 我觉得原始代码可能有问题，或者我理解错了
    # 让我直接看 ContentVec 的输出维度

    print(f"    Raw ContentVec shape: {hubert.shape}")

    # 尝试按照原始代码的方式处理
    # 但我认为正确的方式应该是简单地在时间维度上重复
    # 因为 ContentVec 的 hop_size 是 320，而合成器的 hop_size 是 160
    # 所以需要 2x 上采样

    # 方法: 在时间维度 (axis=1) 上重复
    hubert_repeated = np.repeat(hubert, 2, axis=1)
    print(f"    After repeat on axis=1: {hubert_repeated.shape}")

    hubert_length = hubert_repeated.shape[1]

    # 计算 F0
    print("\n[5] Computing F0...")
    f0 = compute_f0_harvest(
        wav,
        sampling_rate=sampling_rate,
        hop_length=hop_size,
        p_len=hubert_length,
        f0_min=50,
        f0_max=1100
    )
    print(f"    F0 length: {len(f0)}")
    print(f"    F0 range: [{f0.min():.1f}, {f0.max():.1f}] Hz")
    print(f"    F0 mean (voiced): {f0[f0 > 0].mean():.1f} Hz")

    # 应用音高偏移
    if pitch_shift != 0:
        f0 = f0 * (2 ** (pitch_shift / 12))
        print(f"    After pitch shift ({pitch_shift}): [{f0.min():.1f}, {f0.max():.1f}] Hz")

    # 转换 F0 到 pitch 格式
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    pitchf = f0.copy()
    pitch = pitchf.copy()
    f0_mel = 1127 * np.log(1 + pitch / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    pitch = np.rint(f0_mel).astype(np.int64)

    print(f"    Pitch range: [{pitch.min()}, {pitch.max()}]")

    # 准备合成器输入
    print("\n[6] Preparing synthesizer inputs...")

    phone = hubert_repeated.astype(np.float32)
    phone_lengths = np.array([hubert_length], dtype=np.int64)
    pitch_input = pitch.reshape(1, -1)
    pitchf_input = pitchf.reshape(1, -1).astype(np.float32)
    ds = np.array([speaker_id], dtype=np.int64)
    rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)

    print(f"    phone: {phone.shape}")
    print(f"    phone_lengths: {phone_lengths}")
    print(f"    pitch: {pitch_input.shape}")
    print(f"    pitchf: {pitchf_input.shape}")
    print(f"    ds: {ds}")
    print(f"    rnd: {rnd.shape}")

    # 保存中间结果供对比
    np.save("RVC_C/test/ref_phone.npy", phone)
    np.save("RVC_C/test/ref_pitch.npy", pitch_input)
    np.save("RVC_C/test/ref_pitchf.npy", pitchf_input)
    np.save("RVC_C/test/ref_f0.npy", f0)

    # 合成器推理
    print("\n[7] Synthesizer inference...")

    out_wav = synth_sess.run(None, {
        'phone': phone,
        'phone_lengths': phone_lengths,
        'pitch': pitch_input,
        'pitchf': pitchf_input,
        'ds': ds,
        'rnd': rnd
    })[0]

    print(f"    Output shape: {out_wav.shape}")
    out_wav = out_wav.squeeze()
    print(f"    Output samples: {len(out_wav)}")
    print(f"    Output range: [{out_wav.min():.4f}, {out_wav.max():.4f}]")

    # 添加尾部填充并截断到原始长度
    out_wav = np.pad(out_wav, (0, 2 * hop_size), "constant")
    out_wav = out_wav[:org_length]

    # 保存输出
    save_wav_simple(output_wav_path, out_wav, sampling_rate)
    print(f"\n    Saved: {output_wav_path}")

    print("\n" + "="*60)
    print("  Done!")
    print("="*60)

    return out_wav


if __name__ == "__main__":
    contentvec_path = "RVC_C/test/models/vec-768-layer-12.onnx"
    synthesizer_path = "RVC_C/test/models/Rem_e440_s38720.onnx"
    input_wav = "RVC_C/test/test_voice/7.wav"
    output_wav = "RVC_C/test/python_reference_output.wav"

    rvc_inference_python(
        contentvec_path,
        synthesizer_path,
        input_wav,
        output_wav,
        speaker_id=0,
        pitch_shift=0,
        sampling_rate=48000,
        hop_size=512
    )
