# RVC_ONNX - RVC C语言实现

基于 ONNX Runtime 的 RVC (Retrieval-based Voice Conversion) C语言实现。

## 项目概述

本项目将 RVC 从 Python/PyTorch 移植为 C/C++ 实现，使用 ONNX Runtime 进行神经网络推理，实现高性能、低延迟的实时语音转换。

## 特性

- 基于 ONNX Runtime 的神经网络推理
- 支持 Harvest 和 DIO F0 提取算法 (World Vocoder)
- 实时流式音频处理
- 跨平台支持 (Windows/Linux)
- 低延迟音频处理

## 目录结构

```
RVC_C/
├── include/                 # 头文件
│   ├── rvc_onnx.h          # 主API
│   ├── onnx_inference.h    # ONNX推理封装
│   ├── audio_processor.h   # 音频处理
│   ├── f0_extractor.h      # F0提取
│   ├── stft.h              # STFT变换
│   └── utils.h             # 工具函数
├── src/                     # 源文件
│   ├── rvc_onnx.cpp
│   ├── onnx_inference.cpp
│   ├── audio_processor.cpp
│   ├── f0_extractor.cpp
│   ├── stft.cpp
│   └── utils.cpp
├── tests/                   # 测试代码
│   ├── test_onnx_runtime.cpp
│   ├── test_stft.cpp
│   ├── test_f0.cpp
│   └── test_pipeline.cpp
├── examples/                # 示例代码
├── models/                  # ONNX模型文件
├── third_party/             # 第三方库
│   ├── onnx_lib/           # ONNX Runtime 预编译库
│   ├── kissfft/            # KissFFT
│   ├── World/              # World Vocoder
│   ├── libsndfile/         # 音频I/O
│   ├── portaudio/          # 实时音频
│   └── OpenBLAS/           # 数学库
├── documents/               # 文档
└── CMakeLists.txt          # 构建配置
```

## 依赖库

| 库名称 | 版本 | 用途 |
|--------|------|------|
| ONNX Runtime | 1.23.2 | 神经网络推理 |
| KissFFT | - | FFT变换 |
| World Vocoder | - | F0提取 |
| libsndfile | - | 音频文件I/O |
| PortAudio | - | 实时音频 |
| OpenBLAS | - | 矩阵运算 |

## 构建

### 前置条件

- CMake 3.15+
- C++17 兼容编译器
  - Windows: Visual Studio 2019+
  - Linux: GCC 9+

### 编译步骤

```bash
# 创建构建目录
mkdir build
cd build

# 配置 (Windows)
cmake .. -G "Visual Studio 16 2019" -A x64

# 配置 (Linux)
cmake ..

# 编译
cmake --build . --config Release

# 运行测试
./bin/test_onnx_runtime
./bin/test_stft
./bin/test_f0
./bin/test_pipeline
```

## 使用方法

### 基本用法

```c
#include "rvc_onnx.h"

int main() {
    // 创建配置
    RVCConfig config = rvc_default_config();
    config.hubert_model_path = "models/hubert.onnx";
    config.synthesizer_model_path = "models/synthesizer.onnx";
    config.pitch_shift = 0.0f;  // 音高偏移（半音）
    config.num_threads = 4;

    // 创建上下文
    RVCContext* ctx = rvc_create(&config);
    if (!ctx) {
        printf("Failed to create RVC context\n");
        return -1;
    }

    // 转换音频
    float* input_audio = ...;  // 输入音频数据
    size_t input_samples = ...;
    float* output_audio = malloc(input_samples * 3 * sizeof(float));
    size_t output_samples = input_samples * 3;

    RVCError err = rvc_convert(ctx, input_audio, input_samples,
                               output_audio, &output_samples);

    if (err == RVC_SUCCESS) {
        // 使用转换后的音频
    }

    // 清理
    free(output_audio);
    rvc_destroy(ctx);

    return 0;
}
```

### 参数调整

```c
// 设置音高偏移（半音）
rvc_set_pitch_shift(ctx, 5.0f);  // 上移5个半音

// 设置索引率
rvc_set_index_rate(ctx, 0.5f);

// 切换F0提取方法
rvc_set_f0_method(ctx, RVC_F0_DIO);
```

## API 参考

### 主要函数

| 函数 | 描述 |
|------|------|
| `rvc_default_config()` | 获取默认配置 |
| `rvc_create()` | 创建RVC上下文 |
| `rvc_destroy()` | 销毁RVC上下文 |
| `rvc_convert()` | 转换音频 |
| `rvc_stream_convert()` | 流式转换 |
| `rvc_set_pitch_shift()` | 设置音高偏移 |
| `rvc_set_index_rate()` | 设置索引率 |
| `rvc_set_f0_method()` | 设置F0提取方法 |

### 错误码

| 错误码 | 描述 |
|--------|------|
| `RVC_SUCCESS` | 成功 |
| `RVC_ERROR_INVALID_PARAM` | 无效参数 |
| `RVC_ERROR_MODEL_LOAD` | 模型加载失败 |
| `RVC_ERROR_INFERENCE` | 推理失败 |
| `RVC_ERROR_AUDIO_PROCESS` | 音频处理失败 |
| `RVC_ERROR_MEMORY` | 内存分配失败 |

## ONNX 模型准备

需要将 PyTorch 模型转换为 ONNX 格式：

```python
import torch

# HuBERT 模型导出
torch.onnx.export(
    hubert_model,
    dummy_input,
    "hubert.onnx",
    opset_version=11,
    input_names=['audio_input'],
    output_names=['features'],
    dynamic_axes={'audio_input': {1: 'length'}}
)

# 合成器模型导出
torch.onnx.export(
    synthesizer_model,
    (features, f0, sid),
    "synthesizer.onnx",
    opset_version=11,
    input_names=['features', 'f0', 'sid'],
    output_names=['audio']
)
```

## 性能

| 指标 | 目标值 |
|------|--------|
| 延迟 | < 200ms |
| 内存占用 | < 500MB |
| CPU使用率 | < 50% (4核) |

## 开发状态

- [x] 项目结构搭建
- [x] ONNX Runtime 集成
- [x] 音频处理模块
- [x] STFT/Mel 频谱
- [x] F0 提取 (Harvest/DIO)
- [ ] HuBERT 特征提取
- [ ] 合成器推理
- [ ] 实时流式处理
- [ ] 性能优化

## 许可证

MIT License

## 参考

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [ONNX Runtime](https://onnxruntime.ai/)
- [World Vocoder](https://github.com/mmorise/World)
