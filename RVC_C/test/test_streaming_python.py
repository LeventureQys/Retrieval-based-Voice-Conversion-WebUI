"""
RVC 流式语音转换测试脚本
基于 rtrvc.py 的简化实现
"""

import os
import sys
import struct
import numpy as np
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

import onnxruntime as ort
import pyworld
import torch
import torch.nn.functional as F


def load_wav_simple(filepath):
    """简单的 WAV 文件加载"""
    with open(filepath, 'rb') as f:
        f.read(4)  # RIFF
        f.read(4)  # file_size
        f.read(4)  # WAVE

        sample_rate = 16000
        audio_data = None

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                f.read(2)  # audio_format
                channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                f.read(4)  # byte_rate
                f.read(2)  # block_align
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
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))
        f.write(struct.pack('<H', 4))
        f.write(struct.pack('<H', 32))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio.astype(np.float32).tobytes())


def resample_audio(audio, src_rate, dst_rate):
    """重采样音频"""
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate)
    except ImportError:
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


class StreamingRVC:
    """流式 RVC 处理器"""

    def __init__(self, contentvec_path, synthesizer_path, device='cpu'):
        self.device = device
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # 加载模型
        print("Loading ContentVec model...")
        self.cv_sess = ort.InferenceSession(contentvec_path, providers=['CPUExecutionProvider'])
        print("Loading Synthesizer model...")
        self.synth_sess = ort.InferenceSession(synthesizer_path, providers=['CPUExecutionProvider'])

        # 初始化 pitch 缓存 (参考 rtrvc.py)
        self.cache_pitch = torch.zeros(1024, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, dtype=torch.float32)

    def get_f0_post(self, f0):
        """F0 后处理: 转换为 pitch 格式"""
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().squeeze()

        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0_harvest(self, x, f0_up_key):
        """使用 Harvest 提取 F0"""
        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        f0, t = pyworld.harvest(
            x_np.astype(np.double),
            fs=16000,
            f0_ceil=1100,
            f0_floor=50,
            frame_period=10,
        )
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer_chunk(
        self,
        input_wav: torch.Tensor,
        block_frame_16k: int,
        skip_head: int,
        return_length: int,
        f0_up_key: float = 0,
    ) -> np.ndarray:
        """
        流式推理单个块

        参数:
            input_wav: 输入音频 (包含上下文)
            block_frame_16k: 当前块的长度 (不含上下文)
            skip_head: 跳过的输出帧数
            return_length: 返回的样本数
            f0_up_key: 音高偏移 (半音)
        """
        t1 = time.time()

        # 1. ContentVec 推理
        feats_np = input_wav.numpy().reshape(1, 1, -1).astype(np.float32)
        feats = self.cv_sess.run(None, {self.cv_sess.get_inputs()[0].name: feats_np})[0]
        feats = torch.from_numpy(feats)
        # 复制最后一帧
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        t2 = time.time()

        # 2. F0 提取
        p_len = input_wav.shape[0] // 160
        f0_extractor_frame = block_frame_16k + 800

        pitch, pitchf = self.get_f0_harvest(
            input_wav[-f0_extractor_frame:], f0_up_key
        )

        # 3. 更新 pitch 缓存
        shift = block_frame_16k // 160
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        self.cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
        self.cache_pitchf[4 - pitch.shape[0]:] = pitchf[3:-1]

        cache_pitch = self.cache_pitch[None, -p_len:]
        cache_pitchf = self.cache_pitchf[None, -p_len:]

        t3 = time.time()

        # 4. 特征 2x 上采样
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]

        # 5. 准备合成器输入
        phone = feats.numpy().astype(np.float32)
        phone_lengths = np.array([p_len], dtype=np.int64)
        pitch_input = cache_pitch.numpy()
        pitchf_input = cache_pitchf.numpy().astype(np.float32)
        ds = np.array([0], dtype=np.int64)
        rnd = np.random.randn(1, 192, p_len).astype(np.float32)

        # 6. 合成器推理
        out_wav = self.synth_sess.run(None, {
            'phone': phone,
            'phone_lengths': phone_lengths,
            'pitch': pitch_input,
            'pitchf': pitchf_input,
            'ds': ds,
            'rnd': rnd
        })[0]

        t4 = time.time()

        # 7. 提取输出
        out_wav = out_wav.squeeze()

        # 跳过 skip_head 对应的样本
        skip_samples = skip_head * 480  # 48kHz 的 hop_size
        out_wav = out_wav[skip_samples:skip_samples + return_length]

        print(f"  ContentVec: {(t2-t1)*1000:.0f}ms, F0: {(t3-t2)*1000:.0f}ms, Synth: {(t4-t3)*1000:.0f}ms")

        return out_wav


def test_streaming_rvc(
    contentvec_path,
    synthesizer_path,
    input_wav_path,
    output_wav_path,
    block_size_ms=500,
    f0_up_key=0,
):
    """测试流式 RVC"""
    print("="*60)
    print("  RVC Python Streaming Test")
    print("="*60)

    # 加载音频
    print("\n[1] Loading audio...")
    wav, sr = load_wav_simple(input_wav_path)
    print(f"    Loaded: {len(wav)} samples, {sr} Hz")

    # 重采样到 16kHz
    if sr != 16000:
        wav = resample_audio(wav, sr, 16000)
        print(f"    Resampled to 16kHz: {len(wav)} samples")

    wav_tensor = torch.from_numpy(wav).float()

    # 创建流式处理器
    print("\n[2] Creating streaming processor...")
    processor = StreamingRVC(contentvec_path, synthesizer_path)

    # 流式处理参数
    block_size = int(block_size_ms * 16)  # 16kHz
    context_size = 4800  # 300ms 上下文

    total_samples = len(wav)
    num_chunks = (total_samples + block_size - 1) // block_size

    print(f"\n[3] Processing {num_chunks} chunks (block_size={block_size_ms}ms)...")

    all_output = []
    audio_context = torch.zeros(0)

    total_start = time.time()

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * block_size
        chunk_end = min(chunk_start + block_size, total_samples)
        chunk_samples = chunk_end - chunk_start

        current_chunk = wav_tensor[chunk_start:chunk_end]

        # 拼接上下文
        input_with_context = torch.cat([audio_context, current_chunk])

        # 计算 skip_head 和 return_length
        if chunk_idx == 0:
            skip_head = 0
        else:
            skip_head = len(audio_context) // 160

        return_length = chunk_samples * 3  # 16kHz -> 48kHz

        print(f"  Chunk {chunk_idx+1}/{num_chunks}: {chunk_samples} samples, skip_head={skip_head}")

        # 处理
        output = processor.infer_chunk(
            input_with_context,
            chunk_samples,
            skip_head,
            return_length,
            f0_up_key
        )

        all_output.append(output)

        # 更新上下文
        if chunk_samples >= context_size:
            audio_context = current_chunk[-context_size:]
        else:
            audio_context = current_chunk

    total_elapsed = time.time() - total_start

    # 合并输出
    final_output = np.concatenate(all_output)

    print(f"\n[4] Results:")
    print(f"    Total time: {total_elapsed*1000:.0f} ms")
    print(f"    Audio duration: {total_samples/16000:.2f} s")
    print(f"    RTF: {total_elapsed / (total_samples/16000):.2f}")
    print(f"    Output samples: {len(final_output)}")

    # 保存输出
    save_wav_simple(output_wav_path, final_output, 48000)
    print(f"    Saved: {output_wav_path}")

    print("\n" + "="*60)
    print("  Done!")
    print("="*60)

    return final_output


if __name__ == "__main__":
    contentvec_path = "RVC_C/test/models/vec-768-layer-12.onnx"
    synthesizer_path = "RVC_C/test/models/Rem_e440_s38720.onnx"
    input_wav = "RVC_C/test/debug_audio_16k.wav"
    output_wav = "RVC_C/test/python_streaming_output.wav"

    test_streaming_rvc(
        contentvec_path,
        synthesizer_path,
        input_wav,
        output_wav,
        block_size_ms=500,
        f0_up_key=0,
    )
