"""
导出 RVC 推理过程中各层的中间输出，用于与 C++ 实现进行对比
"""

import os
import sys
import numpy as np
import librosa
import onnxruntime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def save_binary(filepath, data, dtype=np.float32):
    """保存数据为二进制格式，包含形状信息"""
    data = np.asarray(data, dtype=dtype)
    with open(filepath, 'wb') as f:
        # 写入维度数
        f.write(np.array([len(data.shape)], dtype=np.uint32).tobytes())
        # 写入每个维度
        for dim in data.shape:
            f.write(np.array([dim], dtype=np.uint64).tobytes())
        # 写入数据
        f.write(data.tobytes())
    print(f"  Saved: {filepath} shape={data.shape} dtype={dtype}")

def load_binary(filepath, dtype=np.float32):
    """加载二进制数据"""
    with open(filepath, 'rb') as f:
        num_dims = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        shape = []
        for _ in range(num_dims):
            dim = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            shape.append(int(dim))
        data = np.frombuffer(f.read(), dtype=dtype).reshape(shape)
    return data

class LayerExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_full_pipeline(self,
                             audio_path,
                             contentvec_path,
                             synthesizer_path,
                             sampling_rate=40000,
                             hop_size=512,
                             f0_up_key=0,
                             sid=0):
        """
        导出完整推理流程的所有中间输出
        """
        print("=" * 60)
        print("RVC Layer Output Exporter")
        print("=" * 60)

        # =====================================================================
        # 1. 加载原始音频
        # =====================================================================
        print("\n[1] Loading audio...")
        wav_original, sr_original = librosa.load(audio_path, sr=None)
        print(f"  Original: {len(wav_original)} samples @ {sr_original} Hz")
        save_binary(f"{self.output_dir}/01_audio_original.bin", wav_original)

        # =====================================================================
        # 2. 重采样到目标采样率
        # =====================================================================
        print("\n[2] Resampling to target sample rate...")
        wav_target_sr = librosa.resample(wav_original, orig_sr=sr_original, target_sr=sampling_rate)
        print(f"  Resampled to {sampling_rate} Hz: {len(wav_target_sr)} samples")
        save_binary(f"{self.output_dir}/02_audio_resampled_{sampling_rate}.bin", wav_target_sr)

        # =====================================================================
        # 3. 重采样到 16kHz (ContentVec 输入)
        # =====================================================================
        print("\n[3] Resampling to 16kHz for ContentVec...")
        wav_16k = librosa.resample(wav_target_sr, orig_sr=sampling_rate, target_sr=16000)
        print(f"  16kHz audio: {len(wav_16k)} samples")
        save_binary(f"{self.output_dir}/03_audio_16k.bin", wav_16k)

        # =====================================================================
        # 4. ContentVec 特征提取
        # =====================================================================
        print("\n[4] Extracting ContentVec features...")
        providers = ["CPUExecutionProvider"]
        contentvec_session = onnxruntime.InferenceSession(contentvec_path, providers=providers)

        # ContentVec 输入: [1, 1, samples]
        contentvec_input = np.expand_dims(np.expand_dims(wav_16k, 0), 0).astype(np.float32)
        print(f"  ContentVec input shape: {contentvec_input.shape}")
        save_binary(f"{self.output_dir}/04_contentvec_input.bin", contentvec_input)

        # 运行 ContentVec
        input_name = contentvec_session.get_inputs()[0].name
        contentvec_output = contentvec_session.run(None, {input_name: contentvec_input})[0]
        print(f"  ContentVec raw output shape: {contentvec_output.shape}")
        save_binary(f"{self.output_dir}/05_contentvec_output_raw.bin", contentvec_output)

        # 转置: [1, 768, frames] -> [1, frames, 768]
        contentvec_transposed = contentvec_output.transpose(0, 2, 1)
        print(f"  ContentVec transposed shape: {contentvec_transposed.shape}")
        save_binary(f"{self.output_dir}/06_contentvec_transposed.bin", contentvec_transposed)

        # =====================================================================
        # 5. 特征重复 (2x) 并转置
        # =====================================================================
        print("\n[5] Repeating features 2x...")
        # repeat along axis=2 (768 dim), then transpose
        hubert = np.repeat(contentvec_transposed, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]
        print(f"  HuBERT features shape: {hubert.shape} (frames={hubert_length})")
        save_binary(f"{self.output_dir}/07_hubert_features.bin", hubert)

        # =====================================================================
        # 6. F0 提取
        # =====================================================================
        print("\n[6] Extracting F0...")
        from infer.lib.infer_pack.modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor

        f0_predictor = HarvestF0Predictor(hop_length=hop_size, sampling_rate=sampling_rate)
        pitchf = f0_predictor.compute_f0(wav_target_sr, hubert_length)
        print(f"  F0 shape: {pitchf.shape}, range: [{pitchf.min():.1f}, {pitchf.max():.1f}] Hz")
        save_binary(f"{self.output_dir}/08_f0_raw.bin", pitchf)

        # =====================================================================
        # 7. F0 音高偏移
        # =====================================================================
        print("\n[7] Applying pitch shift...")
        pitchf_shifted = pitchf * (2 ** (f0_up_key / 12))
        print(f"  Pitch shift: {f0_up_key} semitones")
        print(f"  F0 shifted range: [{pitchf_shifted.min():.1f}, {pitchf_shifted.max():.1f}] Hz")
        save_binary(f"{self.output_dir}/09_f0_shifted.bin", pitchf_shifted)

        # =====================================================================
        # 8. F0 转换为 Mel 刻度
        # =====================================================================
        print("\n[8] Converting F0 to Mel scale...")
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        pitch = pitchf_shifted.copy()
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch_mel = np.rint(f0_mel).astype(np.int64)

        print(f"  Mel pitch range: [{pitch_mel.min()}, {pitch_mel.max()}]")
        save_binary(f"{self.output_dir}/10_pitch_mel.bin", pitch_mel, dtype=np.int64)

        # =====================================================================
        # 9. 准备合成器输入
        # =====================================================================
        print("\n[9] Preparing synthesizer inputs...")

        # phone: [1, frames, 768]
        phone = hubert
        print(f"  phone shape: {phone.shape}")
        save_binary(f"{self.output_dir}/11_synth_phone.bin", phone)

        # phone_lengths: [1]
        phone_lengths = np.array([hubert_length], dtype=np.int64)
        print(f"  phone_lengths: {phone_lengths}")
        save_binary(f"{self.output_dir}/12_synth_phone_lengths.bin", phone_lengths, dtype=np.int64)

        # pitch: [1, frames]
        pitch_input = pitch_mel.reshape(1, -1)
        print(f"  pitch shape: {pitch_input.shape}")
        save_binary(f"{self.output_dir}/13_synth_pitch.bin", pitch_input, dtype=np.int64)

        # pitchf: [1, frames]
        pitchf_input = pitchf_shifted.reshape(1, -1).astype(np.float32)
        print(f"  pitchf shape: {pitchf_input.shape}")
        save_binary(f"{self.output_dir}/14_synth_pitchf.bin", pitchf_input)

        # ds (speaker id): [1]
        ds = np.array([sid], dtype=np.int64)
        print(f"  ds (speaker id): {ds}")
        save_binary(f"{self.output_dir}/15_synth_ds.bin", ds, dtype=np.int64)

        # rnd (random noise): [1, 192, frames]
        # 使用固定种子以便复现
        np.random.seed(42)
        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        print(f"  rnd shape: {rnd.shape}")
        save_binary(f"{self.output_dir}/16_synth_rnd.bin", rnd)

        # =====================================================================
        # 10. 运行合成器
        # =====================================================================
        print("\n[10] Running synthesizer...")
        synth_session = onnxruntime.InferenceSession(synthesizer_path, providers=providers)

        input_names = [inp.name for inp in synth_session.get_inputs()]
        print(f"  Synthesizer input names: {input_names}")

        synth_inputs = {
            input_names[0]: phone,
            input_names[1]: phone_lengths,
            input_names[2]: pitch_input,
            input_names[3]: pitchf_input,
            input_names[4]: ds,
            input_names[5]: rnd,
        }

        synth_output = synth_session.run(None, synth_inputs)[0]
        print(f"  Synthesizer output shape: {synth_output.shape}")
        save_binary(f"{self.output_dir}/17_synth_output.bin", synth_output)

        # =====================================================================
        # 11. 保存最终音频
        # =====================================================================
        print("\n[11] Saving final audio...")
        output_wav = synth_output.squeeze()

        # 保存为 WAV
        import soundfile as sf
        wav_path = f"{self.output_dir}/output_python.wav"
        sf.write(wav_path, output_wav, sampling_rate)
        print(f"  Saved: {wav_path}")

        # =====================================================================
        # 12. 导出兼容旧格式的文件 (用于 test_synth_compare)
        # =====================================================================
        print("\n[12] Exporting legacy format files...")
        save_binary(f"{self.output_dir}/bin_phone.bin", phone)
        save_binary(f"{self.output_dir}/bin_pitch.bin", pitch_input, dtype=np.int64)
        save_binary(f"{self.output_dir}/bin_pitchf.bin", pitchf_input)
        save_binary(f"{self.output_dir}/bin_rnd.bin", rnd)
        save_binary(f"{self.output_dir}/bin_output_python.bin", synth_output)

        print("\n" + "=" * 60)
        print("Export completed!")
        print("=" * 60)

        return {
            'hubert_length': hubert_length,
            'sampling_rate': sampling_rate,
            'hop_size': hop_size,
        }


def compare_with_cpp(output_dir, cpp_output_path):
    """
    比较 Python 和 C++ 的输出
    """
    print("\n" + "=" * 60)
    print("Comparing Python vs C++ outputs")
    print("=" * 60)

    py_output = load_binary(f"{output_dir}/17_synth_output.bin")

    # 加载 C++ 输出 (假设是 WAV 文件)
    import soundfile as sf
    cpp_output, sr = sf.read(cpp_output_path)

    min_len = min(len(py_output.flatten()), len(cpp_output.flatten()))
    py_flat = py_output.flatten()[:min_len]
    cpp_flat = cpp_output.flatten()[:min_len]

    diff = np.abs(py_flat - cpp_flat)

    print(f"\nComparison results (first {min_len} samples):")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Std diff: {diff.std():.6f}")
    print(f"  Samples with diff > 0.01: {(diff > 0.01).sum()} ({100*(diff > 0.01).mean():.2f}%)")
    print(f"  Samples with diff > 0.1: {(diff > 0.1).sum()} ({100*(diff > 0.1).mean():.2f}%)")

    # 计算相关系数
    correlation = np.corrcoef(py_flat, cpp_flat)[0, 1]
    print(f"  Correlation: {correlation:.6f}")


if __name__ == "__main__":
    # 配置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "layer_outputs")

    # 模型路径
    contentvec_path = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/assets/hubert/vec-768-layer-12.onnx"
    synthesizer_path = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/models/Rem_e440_s38720.onnx"

    # 输入音频路径
    audio_path = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/test_input.wav"

    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        print("Please provide a test audio file.")
        sys.exit(1)

    if not os.path.exists(contentvec_path):
        print(f"Error: ContentVec model not found: {contentvec_path}")
        sys.exit(1)

    if not os.path.exists(synthesizer_path):
        print(f"Error: Synthesizer model not found: {synthesizer_path}")
        sys.exit(1)

    # 运行导出
    exporter = LayerExporter(output_dir)
    result = exporter.export_full_pipeline(
        audio_path=audio_path,
        contentvec_path=contentvec_path,
        synthesizer_path=synthesizer_path,
        sampling_rate=40000,
        hop_size=512,
        f0_up_key=0,
        sid=0
    )

    print(f"\nAll outputs saved to: {output_dir}")
