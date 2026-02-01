# RVC ONNX版本C语言开发计划

## 1. 项目概述

### 1.1 目标
将RVC（Retrieval-based Voice Conversion）从Python/PyTorch版本移植为基于ONNX Runtime的C语言版本，实现高性能、低延迟的实时语音转换。

### 1.2 核心特性
- 基于ONNX Runtime的神经网络推理
- 实时流式音频处理
- 支持v1/v2模型架构
- CPU推理优化
- 低延迟音频处理

## 2. 第三方库需求

### 2.1 必需依赖库

| 库名称 | 版本要求 | 用途 | 许可证 | 安装方式 |
|--------|----------|------|--------|----------|
| **ONNX Runtime** | ≥1.16.0 | 神经网络推理引擎 | MIT | 预编译包/源码编译 |
| **libsndfile** | ≥1.0.30 | 音频文件I/O | LGPL/GPL | 包管理器/源码编译 |
| **FFTW3** | ≥3.3.10 | FFT变换 | GPL/BSD | 包管理器/源码编译 |
| **PortAudio** | ≥19.7 | 实时音频设备访问 | MIT | 包管理器/源码编译 |
| **CMake** | ≥3.15 | 构建系统 | BSD | 包管理器 |

### 2.2 可选依赖库

| 库名称 | 版本要求 | 用途 | 许可证 | 说明 |
|--------|----------|------|--------|------|
| **Intel MKL** | ≥2023 | 高性能数学库 | Proprietary | 性能优化 |
| **OpenMP** | - | 多线程并行 | - | CPU并行计算 |
| **FAISS** | ≥1.7.0 | 向量检索 | MIT | 可选功能 |
| **libsamplerate** | ≥0.2.2 | 音频重采样 | BSD | 高质量重采样 |

### 2.3 平台特定依赖

**Windows:**
- Visual Studio 2019或更高版本
- vcpkg包管理器（推荐）

**Linux:**
- GCC 9.0或更高版本
- pkg-config

**macOS:**
- Xcode 12.0或更高版本
- Homebrew包管理器

## 3. 开发者准备工作

### 3.1 环境准备

#### 3.1.1 开发环境
```bash
# Windows (使用vcpkg)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install

# 安装依赖
./vcpkg install onnxruntime-static onnxruntime-shared libsndfile fftw3 portaudio

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install cmake build-essential
sudo apt install libsndfile1-dev libfftw3-dev portaudio19-dev
```

#### 3.1.2 ONNX模型准备
需要将现有的PyTorch模型转换为ONNX格式：

```python
# 示例：HuBERT模型转换
import torch
import onnx

# 加载PyTorch模型
hubert_model = load_hubert_model()

# 准备示例输入
dummy_input = torch.randn(1, 16000)  # 1秒音频

# 导出ONNX模型
torch.onnx.export(
    hubert_model,
    dummy_input,
    "hubert.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {1: 'sequence'},
        'output': {1: 'sequence'}
    }
)
```

### 3.2 项目结构准备

```
RVC_ONNX/
├── include/                 # 头文件
│   ├── rvc_onnx.h          # 主API
│   ├── audio_processor.h   # 音频处理
│   ├── onnx_inference.h    # ONNX推理封装
│   ├── f0_extractor.h      # F0提取
│   └── utils.h             # 工具函数
├── src/                     # 源文件
│   ├── rvc_onnx.cpp
│   ├── audio_processor.cpp
│   ├── onnx_inference.cpp
│   ├── f0_extractor.cpp
│   └── utils.cpp
├── models/                  # ONNX模型文件
│   ├── hubert.onnx
│   ├── synthesizer.onnx
│   └── rmvpe.onnx
├── third_party/             # 第三方库（可选）
├── examples/                # 示例代码
│   ├── simple_convert.cpp
│   └── realtime_demo.cpp
├── tests/                   # 测试代码
├── CMakeLists.txt           # 构建配置
└── README.md
```

### 3.3 开发工具准备

- **IDE**: VS Code + C/C++扩展，或 CLion
- **调试器**: GDB (Linux/macOS), Visual Studio Debugger (Windows)
- **性能分析**: Valgrind, Intel VTune, 或 perf
- **版本控制**: Git

## 4. 开发流程

### 4.1 阶段一：基础设施搭建 (Week 1-2)

#### 任务1.1：项目结构搭建
- [ ] 创建项目目录结构
- [ ] 配置CMakeLists.txt
- [ ] 设置Git仓库和分支策略
- [ ] 配置CI/CD基础框架

#### 任务1.2：依赖库集成
- [ ] 集成ONNX Runtime C++ API
- [ ] 集成音频I/O库 (libsndfile)
- [ ] 集成信号处理库 (FFTW3)
- [ ] 编写基础测试验证依赖

#### 任务1.3：构建系统配置
```cmake
# CMakeLists.txt 示例
cmake_minimum_required(VERSION 3.15)
project(RVC_ONNX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED onnxruntime)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FFTW3 REQUIRED fftw3)

add_executable(rvc_onnx
    src/main.cpp
    src/rvc_onnx.cpp
    # ... 其他源文件
)

target_link_libraries(rvc_onnx
    ${ONNXRUNTIME_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    ${FFTW3_LIBRARIES}
)
```

### 4.2 阶段二：核心模块开发 (Week 3-6)

#### 任务2.1：ONNX推理引擎封装
- [ ] 设计ONNXModel基类
- [ ] 实现HuBERT模型推理接口
- [ ] 实现RVC合成器模型推理接口
- [ ] 实现RMVPE模型推理接口（可选）
- [ ] 添加模型加载和缓存机制

```cpp
// 示例：ONNX模型封装
class ONNXModel {
public:
    ONNXModel(const std::string& model_path);
    std::vector<float> run_inference(const std::vector<float>& input);
    void set_threads(int threads);
    
private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
};
```

#### 任务2.2：音频处理模块
- [ ] 实现音频缓冲区管理
- [ ] 实现重采样功能
- [ ] 实现音频预处理（归一化、静音检测）
- [ ] 实现STFT/ISTFT变换

#### 任务2.3：F0提取模块
- [ ] 集成Harvest算法（C++版本）
- [ ] 实现F0后处理（平滑、插值）
- [ ] 添加F0转换功能（音高调整）

#### 任务2.4：特征处理模块
- [ ] 实现HuBERT特征提取和处理
- [ ] 实现特征插值和对齐
- [ ] 实现FAISS检索接口（可选）

### 4.3 阶段三：流式处理实现 (Week 7-9)

#### 任务3.1：实时音频处理
- [ ] 实现音频流缓冲机制
- [ ] 实现块处理和重叠处理
- [ ] 实现低延迟音频I/O

#### 任务3.2：同步和时序管理
- [ ] 实现多线程同步机制
- [ ] 实现时序对齐算法
- [ ] 实现延迟测量和优化

#### 任务3.3：内存管理优化
- [ ] 实现内存池管理
- [ ] 优化内存分配和释放
- [ ] 实现零拷贝数据传输

### 4.4 阶段四：集成和优化 (Week 10-12)

#### 任务4.1：模块集成
- [ ] 集成所有核心模块
- [ ] 实现完整的推理管线
- [ ] 添加错误处理和恢复机制

#### 任务4.2：性能优化
- [ ] CPU性能分析和优化
- [ ] 内存使用优化
- [ ] 并行处理优化

#### 任务4.3：API设计和完善
- [ ] 设计简洁的C API
- [ ] 实现高级功能接口
- [ ] 添加配置和参数管理

### 4.5 阶段五：测试和验证 (Week 13-14)

#### 任务5.1：功能测试
- [ ] 与Python版本输出对比测试
- [ ] 边界条件测试
- [ ] 长时间运行稳定性测试

#### 任务5.2：性能测试
- [ ] 延迟性能测试
- [ ] 内存使用测试
- [ ] CPU使用率测试

#### 任务5.3：兼容性测试
- [ ] 多平台兼容性测试
- [ ] 不同模型兼容性测试
- [ ] 音频格式兼容性测试

## 5. 核心开发指导

### 5.1 ONNX模型转换指南

#### 5.1.1 HuBERT模型转换
```python
def export_hubert_onnx():
    # 加载预训练模型
    hubert = torch.hub.load('facebookresearch/fairseq', 'hubert_base')
    hubert.eval()
    
    # 准备示例输入
    dummy_input = torch.randn(1, 16000)  # 1秒16kHz音频
    
    # 导出ONNX
    torch.onnx.export(
        hubert,
        dummy_input,
        "hubert.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,  # 优化常量折叠
        input_names=['audio_input'],
        output_names=['features'],
        dynamic_axes={
            'audio_input': {1: 'audio_length'},
            'features': {1: 'feature_length'}
        }
    )
```

#### 5.1.2 RVC合成器模型转换
```python
def export_synthesizer_onnx():
    # 加载RVC模型
    model = SynthesizerTrnMs256NSFsid(...)
    model.eval()
    
    # 准备示例输入
    features = torch.randn(1, 256, 100)  # HuBERT特征
    f0 = torch.randn(1, 1, 100)         # F0特征
    sid = torch.LongTensor([0])          # 说话人ID
    
    # 导出ONNX
    torch.onnx.export(
        model,
        (features, f0, sid),
        "synthesizer.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['features', 'f0', 'sid'],
        output_names=['audio_output']
    )
```

### 5.2 C++ ONNX推理实现

#### 5.2.1 基础推理类
```cpp
#include <onnxruntime_cxx_api.h>

class RVCOnnxInference {
public:
    RVCOnnxInference(const std::string& hubert_model_path,
                     const std::string& synth_model_path);
    
    // 推理函数
    std::vector<float> convert_audio(const std::vector<float>& input_audio,
                                   float pitch_shift = 0.0f);
    
private:
    Ort::Env env_{nullptr};
    Ort::SessionOptions session_options_{nullptr};
    
    // 模型实例
    std::unique_ptr<Ort::Session> hubert_session_;
    std::unique_ptr<Ort::Session> synth_session_;
    
    // 推理辅助函数
    std::vector<float> extract_hubert_features(const std::vector<float>& audio);
    std::vector<float> synthesize_audio(const std::vector<float>& features,
                                      const std::vector<float>& f0);
};
```

#### 5.2.2 音频处理实现
```cpp
class AudioProcessor {
public:
    // 重采样
    std::vector<float> resample(const std::vector<float>& input, 
                               int src_rate, int dst_rate);
    
    // STFT变换
    std::vector<std::complex<float>> stft(const std::vector<float>& audio,
                                        int frame_size, int hop_size);
    
    // 音频块处理
    void process_block(const float* input, float* output, int block_size);
    
private:
    // 内部缓冲区和状态
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
};
```

### 5.3 性能优化策略

#### 5.3.1 内存优化
- 使用内存池减少动态分配
- 预分配固定大小的缓冲区
- 重用中间计算结果

#### 5.3.2 计算优化
- 启用ONNX Runtime优化选项
- 使用多线程推理
- 实现SIMD优化

#### 5.3.3 延迟优化
- 实现流式处理
- 优化块大小和重叠
- 减少不必要的数据拷贝

### 5.4 错误处理和日志

```cpp
// 错误处理宏
#define ORT_ABORT_ON_ERROR(expr) \
    do { \
        OrtStatus* onnx_status = (expr); \
        if (onnx_status != NULL) { \
            const char* msg = OrtGetErrorMessage(onnx_status); \
            fprintf(stderr, "ONNX Runtime error: %s\n", msg); \
            OrtReleaseStatus(onnx_status); \
            abort(); \
        } \
    } while(0)

// 日志系统
enum LogLevel { DEBUG, INFO, WARNING, ERROR };
void log_message(LogLevel level, const char* fmt, ...);
```

## 6. 测试验证策略

### 6.1 单元测试
- 模型推理准确性测试
- 音频处理功能测试
- 边界条件测试

### 6.2 集成测试
- 端到端转换测试
- 与Python版本输出对比
- 实时性能测试

### 6.3 性能基准测试
- 延迟测量
- 内存使用监控
- CPU使用率分析

## 7. 部署和分发

### 7.1 构建配置
- 静态链接 vs 动态链接
- 优化级别设置
- 平台特定配置

### 7.2 分发包制作
- 交叉平台构建脚本
- 依赖库打包
- 安装程序制作

## 8. 维护和更新

### 8.1 版本管理
- 语义化版本控制
- 模型版本兼容性
- API向后兼容

### 8.2 文档维护
- API文档
- 使用示例
- 性能基准

---

**项目周期**: 预计14周完成
**团队规模**: 1-2名C++开发者
**风险评估**: 中等（主要风险在模型转换和性能优化）
**成功标准**: 实时延迟<200ms，内存占用<500MB，输出质量与Python版本一致