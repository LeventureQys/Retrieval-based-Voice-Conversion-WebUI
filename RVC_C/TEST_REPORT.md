# RVC_C 测试报告

## 概述

RVC_C 是 RVC (Retrieval-based Voice Conversion) 的 C/C++ 实现，使用 ONNX Runtime 进行神经网络推理。本报告总结了各模块的测试结果。

## 测试环境

- **平台**: Windows 10/11
- **编译器**: MSVC 19.43 (Visual Studio 2022)
- **ONNX Runtime**: 1.23.2
- **C++ 标准**: C++17
- **C 标准**: C11

## 测试结果

### 1. STFT 模块 ✅ 通过

- 窗函数创建 (Hann, Hamming)
- Hz-Mel 频率转换
- Mel 滤波器组创建
- STFT 正向变换
- STFT 往返测试 (重建误差: 0.000000)

### 2. F0 提取模块 ✅ 通过

- Harvest 算法
- DIO 算法
- 频率-MIDI 转换
- F0 后处理 (平滑、插值、音高偏移)

### 3. 音频处理模块 ✅ 通过

- WAV 文件读取 (16-bit PCM, 24-bit PCM, 32-bit float)
- WAV 文件写入
- 音频重采样 (16kHz ↔ 48kHz)
- 音频归一化
- 静音检测

### 4. ONNX 推理引擎 ✅ 通过

- 引擎创建/销毁
- 模型加载
- 多数据类型支持 (float32, int64, int32)
- 多输入多输出推理

### 5. ContentVec 特征提取 ✅ 通过

- 模型加载 (vec-768-layer-12.onnx, 360MB)
- 输入: [1, 1, audio_length] @ 16kHz
- 输出: [1, time_frames, 768]
- 推理时间: ~766 ms (19.6秒音频)

### 6. 合成器推理 ✅ 通过

- 模型加载 (Rem_e440_s38720.onnx, 110MB)
- 6 输入 / 1 输出
- 推理时间: ~6.2 秒 (19.6秒音频)

### 7. 端到端语音转换 ✅ 通过

完整流程测试结果:

| 步骤 | 耗时 |
|------|------|
| ContentVec 加载 | ~0.5s |
| Synthesizer 加载 | ~1s |
| 音频加载 + 重采样 | <0.1s |
| F0 提取 (Harvest) | ~2.6s |
| ContentVec 推理 | ~0.8s |
| 合成器推理 | ~6.2s |
| **总计** | **~11s** |

测试音频: 19.6秒 (942080 samples @ 48kHz)
输出音频: 19.6秒 (941760 samples @ 48kHz)

## 模型信息

### ContentVec (vec-768-layer-12.onnx)

**输入:**
- `source`: [1, 1, audio_length] - 16kHz 音频 (float32)

**输出:**
- `embed`: [1, time_frames, 768] - 语音特征 (float32)

### Synthesizer (Rem_e440_s38720.onnx)

**输入:**
- `phone`: [1, T, 768] - ContentVec 特征 (float32)
- `phone_lengths`: [1] - 序列长度 (int64)
- `pitch`: [1, T] - 量化音高 1-255 (int64)
- `pitchf`: [1, T] - 连续音高 Hz (float32)
- `ds`: [1] - 说话人 ID (int64)
- `rnd`: [1, 192, T] - 随机噪声 (float32)

**输出:**
- `audio`: [1, 1, samples] - 生成的音频波形 (float32)

## 使用方法

### 编译

```bash
cd RVC_C
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### 运行语音转换

```bash
cd build/bin/Release

./test_full_pipeline.exe \
    -c path/to/vec-768-layer-12.onnx \
    -s path/to/synthesizer.onnx \
    -i input.wav \
    -o output.wav \
    -p 0        # 音高偏移 (半音)
    -sid 0      # 说话人 ID
```

### 运行单元测试

```bash
./test_stft.exe          # STFT 测试
./test_f0.exe            # F0 提取测试
./test_pipeline.exe      # 基础流程测试
./test_synthesizer.exe -m model.onnx  # 合成器测试
```

## 文件结构

```
RVC_C/
├── include/           # 头文件
│   ├── rvc_onnx.h
│   ├── onnx_inference.h
│   ├── audio_processor.h
│   ├── f0_extractor.h
│   ├── stft.h
│   └── utils.h
├── src/               # 源文件
│   ├── rvc_onnx.cpp
│   ├── onnx_inference.cpp
│   ├── audio_processor.cpp
│   ├── f0_extractor.cpp
│   ├── stft.cpp
│   └── utils.cpp
├── tests/             # 测试程序
│   ├── test_onnx_runtime.cpp
│   ├── test_stft.cpp
│   ├── test_f0.cpp
│   ├── test_pipeline.cpp
│   ├── test_synthesizer.cpp
│   └── test_full_pipeline.cpp  # 完整端到端测试
├── test/              # 测试数据
│   ├── models/
│   │   ├── vec-768-layer-12.onnx
│   │   └── Rem_e440_s38720.onnx
│   ├── test_voice/
│   │   ├── 7.wav
│   │   └── enrollment_000001.wav
│   ├── converted_output.wav      # 转换输出
│   └── converted_pitch_up.wav    # 升调转换输出
└── third_party/       # 第三方库
    ├── onnx_lib/
    ├── kissfft/
    ├── World/
    └── ...
```

## 性能分析

对于 19.6 秒的输入音频:
- 实时率: ~0.56x (处理时间 / 音频时长)
- 主要瓶颈: 合成器推理 (~56%)

优化建议:
1. 使用 GPU 加速 (CUDA/DirectML)
2. 减少 F0 提取精度 (使用 DIO 替代 Harvest)
3. 批处理优化

## 结论

RVC_C 已实现完整的端到端语音转换功能:

✅ 音频加载和预处理
✅ F0 基频提取
✅ ContentVec 特征提取
✅ 合成器推理
✅ 音高偏移支持
✅ 音频输出

所有核心模块均已通过测试，可以进行实际的语音转换任务。
