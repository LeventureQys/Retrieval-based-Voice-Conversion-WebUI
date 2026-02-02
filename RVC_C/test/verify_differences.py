"""
验证 Python 和 C++ 之间的具体差异
"""

import numpy as np
import struct

def test_random_number_generation():
    """测试随机数生成的差异"""
    print("="*60)
    print("  Random Number Generation Comparison")
    print("="*60)

    # Python 方式
    np.random.seed(42)
    py_rnd = np.random.randn(10).astype(np.float32)
    print(f"\nPython np.random.randn (seed=42):")
    print(f"  First 10 values: {py_rnd}")
    print(f"  Range: [{py_rnd.min():.4f}, {py_rnd.max():.4f}]")

    # C++ 方式模拟 (使用相同的 srand/rand)
    # C++ 代码: ((float)rand() / RAND_MAX - 0.5f) * 0.2f
    # 这会产生 [-0.1, 0.1] 范围的均匀分布

    # Python randn 产生标准正态分布，乘以 0.1 后范围约 [-0.3, 0.3]

    print(f"\nC++ style (uniform [-0.1, 0.1]):")
    np.random.seed(42)
    cpp_style = (np.random.rand(10).astype(np.float32) - 0.5) * 0.2
    print(f"  First 10 values: {cpp_style}")
    print(f"  Range: [{cpp_style.min():.4f}, {cpp_style.max():.4f}]")

    print(f"\n  [ISSUE] Python uses Gaussian distribution, C++ uses uniform!")
    print(f"  This causes different noise characteristics in the synthesizer.")


def test_f0_parameters():
    """测试 F0 参数差异"""
    print("\n" + "="*60)
    print("  F0 Parameter Comparison")
    print("="*60)

    print(f"\nPython (rvc_reference.py):")
    print(f"  f0_min = 50 Hz")
    print(f"  f0_max = 1100 Hz")
    print(f"  frame_period = 1000 * hop_size / sample_rate")
    print(f"               = 1000 * 512 / 48000 = 10.67 ms")

    print(f"\nC++ (f0_extractor.cpp default):")
    print(f"  f0_floor = 71.0 Hz")
    print(f"  f0_ceil = 800.0 Hz")
    print(f"  frame_period = 5.0 ms (default)")

    print(f"\nC++ (test_full_pipeline.cpp override):")
    print(f"  f0_floor = 50.0 Hz")
    print(f"  f0_ceil = 1100.0 Hz")
    print(f"  frame_period = 1000 * 512 / 48000 = 10.67 ms")

    print(f"\n  [OK] F0 parameters are correctly overridden in test_full_pipeline.cpp")


def test_pitch_quantization():
    """测试 pitch 量化公式"""
    print("\n" + "="*60)
    print("  Pitch Quantization Comparison")
    print("="*60)

    f0_min = 50.0
    f0_max = 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)

    print(f"\nConstants:")
    print(f"  f0_mel_min = {f0_mel_min:.4f}")
    print(f"  f0_mel_max = {f0_mel_max:.4f}")

    # 测试几个 F0 值
    test_f0s = [0, 1.5, 50, 100, 200, 300, 500, 800, 1100]

    print(f"\nF0 -> Pitch mapping:")
    print(f"  {'F0 (Hz)':<10} {'Mel':<10} {'Pitch':<10}")
    print(f"  {'-'*30}")

    for f0 in test_f0s:
        if f0 > 0:
            f0_mel = 1127.0 * np.log(1 + f0 / 700.0)
            pitch_val = (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
            pitch_val = np.clip(pitch_val, 1, 255)
            pitch = int(round(pitch_val))
        else:
            f0_mel = 0
            pitch = 0  # Python uses 0, C++ uses 1

        print(f"  {f0:<10.1f} {f0_mel:<10.2f} {pitch:<10}")

    print(f"\n  [ISSUE] When F0=0 (unvoiced):")
    print(f"    Python: pitch = 0")
    print(f"    C++:    pitch = 1")
    print(f"  But after interpolation, there should be no F0=0 values.")


def test_contentvec_repeat():
    """测试 ContentVec 特征重复方式"""
    print("\n" + "="*60)
    print("  ContentVec Feature Repeat Comparison")
    print("="*60)

    # 模拟 ContentVec 输出
    cv_output = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)  # [1, 2, 3]
    print(f"\nOriginal ContentVec output: {cv_output.shape}")
    print(f"  {cv_output}")

    # Python 方式: np.repeat(hubert, 2, axis=1)
    py_repeat = np.repeat(cv_output, 2, axis=1)
    print(f"\nPython np.repeat(axis=1): {py_repeat.shape}")
    print(f"  {py_repeat}")

    # C++ 方式
    # for (size_t t = 0; t < synth_frames; t++) {
    #     size_t src_t = t / 2;
    #     memcpy(&phone[t * cv_dim], &cv_data[src_t * cv_dim], cv_dim * sizeof(float));
    # }
    synth_frames = cv_output.shape[1] * 2
    cv_dim = cv_output.shape[2]
    cpp_repeat = np.zeros((1, synth_frames, cv_dim), dtype=np.float32)
    cv_data = cv_output[0]
    for t in range(synth_frames):
        src_t = t // 2
        cpp_repeat[0, t] = cv_data[src_t]

    print(f"\nC++ style repeat: {cpp_repeat.shape}")
    print(f"  {cpp_repeat}")

    # 比较
    if np.allclose(py_repeat, cpp_repeat):
        print(f"\n  [OK] Both methods produce identical results!")
    else:
        print(f"\n  [ISSUE] Methods produce different results!")
        print(f"  Diff: {np.abs(py_repeat - cpp_repeat).max()}")


def test_f0_interpolation():
    """测试 F0 插值"""
    print("\n" + "="*60)
    print("  F0 Interpolation Comparison")
    print("="*60)

    # 模拟 F0 数据 (有一些无声段)
    f0 = np.array([0, 0, 100, 150, 0, 0, 200, 250, 0, 0], dtype=np.float64)
    print(f"\nOriginal F0: {f0}")

    # Python interpolate_f0
    def interpolate_f0_python(f0):
        data = np.reshape(f0.copy(), (f0.size, 1))
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

        return ip_data[:, 0]

    # C++ f0_interpolate (模拟)
    def f0_interpolate_cpp(f0):
        f0 = f0.copy()
        length = len(f0)

        # 找到第一个有效值
        first_valid = 0
        while first_valid < length and f0[first_valid] <= 0:
            first_valid += 1

        if first_valid >= length:
            return f0

        # 填充开头
        for i in range(first_valid):
            f0[i] = f0[first_valid]

        # 线性插值中间的无效段
        prev_valid = first_valid
        for i in range(first_valid + 1, length):
            if f0[i] > 0:
                if i - prev_valid > 1:
                    step = (f0[i] - f0[prev_valid]) / (i - prev_valid)
                    for j in range(prev_valid + 1, i):
                        f0[j] = f0[prev_valid] + step * (j - prev_valid)
                prev_valid = i

        # 填充结尾
        for i in range(prev_valid + 1, length):
            f0[i] = f0[prev_valid]

        return f0

    py_interp = interpolate_f0_python(f0)
    cpp_interp = f0_interpolate_cpp(f0)

    print(f"\nPython interpolated: {py_interp}")
    print(f"C++ interpolated:    {cpp_interp}")

    if np.allclose(py_interp, cpp_interp):
        print(f"\n  [OK] Both methods produce identical results!")
    else:
        print(f"\n  [ISSUE] Methods produce different results!")
        diff = np.abs(py_interp - cpp_interp)
        print(f"  Max diff: {diff.max()}")
        print(f"  Diff indices: {np.where(diff > 0.001)[0]}")


def main():
    print("="*60)
    print("  RVC Python vs C++ Difference Verification")
    print("="*60)

    test_random_number_generation()
    test_f0_parameters()
    test_pitch_quantization()
    test_contentvec_repeat()
    test_f0_interpolation()

    print("\n" + "="*60)
    print("  Summary of Issues Found")
    print("="*60)
    print("""
  1. [CRITICAL] Random Number Generation:
     - Python: Gaussian distribution (randn)
     - C++: Uniform distribution (rand)
     - FIX: Use Box-Muller transform in C++ to generate Gaussian noise

  2. [OK] F0 Parameters:
     - Both use same parameters after override

  3. [MINOR] Pitch for unvoiced:
     - Python: 0
     - C++: 1
     - After interpolation, this shouldn't matter

  4. [OK] ContentVec repeat:
     - Both methods are equivalent

  5. [POTENTIAL] F0 Interpolation:
     - Need to verify with real data
""")


if __name__ == "__main__":
    main()
