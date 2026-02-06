# RVC_C 流式推理实现文档

## 项目概述

本文档记录了为 RVC_C 项目添加流式（实时）语音转换功能的设计与实现进度。

## 背景

### 参考实现

Python 版本的流式推理实现位于 `infer/lib/rtrvc.py`，其核心机制包括：

1. **Pitch 缓存** - 维护 1024 帧历史，保持音高连续性
2. **滑动窗口** - 每次处理后，缓存左移 `block_frame_16k // 160` 帧
3. **skip_head / return_length** - 合成器只返回新生成的部分，避免重复
4. **特征 2x 上采样** - ContentVec 特征插值到合成器帧率

### 帧率对齐

| 组件 | hop_size | 帧率 (16kHz) |
|------|----------|--------------|
| ContentVec | 320 | 50 fps |
| 合成器 | 160 | 100 fps |
| F0 提取 | 160 | 100 fps |

## 设计方案

### 功能范围

- **基础流式推理**（无 FAISS 索引）
- **F0 方法**: 仅 Harvest/DIO（已在 RVC_C 中实现）

### 新增文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/rvc_streaming.h` | ✅ 已完成 | 流式 API 定义 |
| `src/rvc_streaming.cpp` | ✅ 已完成 | 流式处理实现 (基础框架) |
| `tests/test_streaming.cpp` | ✅ 已完成 | 测试程序 (11/11 通过) |

### 需修改文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `CMakeLists.txt` | ✅ 已完成 | 添加新源文件 |
| `src/rvc_onnx.cpp` | ⏳ 待修改 | 集成流式处理 |

## API 设计

### 数据结构

```c
/** 流式处理状态 */
typedef struct RVCStreamState {
    // Pitch 缓存 (1024 帧)
    int64_t* cache_pitch;           // 量化音高缓存
    float* cache_pitchf;            // 连续音高缓存

    // 音频输入缓冲 (用于 F0 提取的额外上下文)
    float* audio_context;           // 前一块的尾部音频
    size_t audio_context_size;      // 音频上下文大小

    // 特征缓冲 (用于重叠处理)
    float* feature_context;         // 前一块的尾部特征
    size_t feature_context_frames;  // 特征上下文帧数

    // 状态标志
    int is_first_chunk;             // 是否第一块
    size_t total_samples_processed; // 已处理的总样本数
} RVCStreamState;

/** 流式配置 */
typedef struct RVCStreamConfig {
    size_t block_size;              // 处理块大小 (默认: 16000)
    size_t audio_context_size;      // 音频上下文大小 (默认: 4800)
    size_t crossfade_samples;       // 交叉淡化样本数 (默认: 480)
    int speaker_id;                 // 说话人 ID
} RVCStreamConfig;
```

### 核心函数

```c
// 创建流式状态
RVCStreamState* rvc_stream_state_create(const RVCStreamConfig* config);

// 销毁流式状态
void rvc_stream_state_destroy(RVCStreamState* state);

// 重置流式状态
void rvc_stream_state_reset(RVCStreamState* state);

// 流式转换 (核心函数)
RVCStreamError rvc_stream_process(
    RVCContext* ctx,
    RVCStreamState* state,
    const float* input_chunk,      // 输入音频块 (16kHz)
    size_t input_samples,          // 输入样本数
    float* output_chunk,           // 输出音频块 (48kHz)
    size_t output_capacity,        // 输出缓冲区容量
    size_t* output_samples,        // 输出样本数
    const RVCStreamConfig* config  // 流式配置
);
```

## 核心算法

### 处理流程

```
输入: input_chunk (16kHz 音频块)
输出: output_chunk (48kHz 转换后音频)

1. 拼接音频上下文
   audio_with_context = [audio_context, input_chunk]

2. 提取 F0
   f0_result = f0_extract(audio_with_context)
   应用音高偏移

3. 提取 ContentVec 特征
   cv_features = contentvec_infer(audio_with_context)

4. 特征 2x 上采样
   synth_features = interpolate_2x(cv_features)

5. 更新 Pitch 缓存
   shift = input_samples / 160
   cache_pitch[:-shift] = cache_pitch[shift:]
   cache_pitch[-new_frames:] = new_pitch

6. 准备合成器输入
   - phone: synth_features
   - pitch: cache_pitch[-p_len:]
   - pitchf: cache_pitchf[-p_len:]
   - skip_head: 重叠帧数
   - return_length: 新生成样本数

7. 运行合成器
   output = synthesizer_infer(inputs)

8. 更新上下文
   audio_context = input_chunk[-context_size:]

9. 返回输出音频
```

### Pitch 缓存更新

```c
void update_pitch_cache(
    RVCStreamState* state,
    const int64_t* new_pitch,
    const float* new_pitchf,
    size_t new_length,
    size_t shift
) {
    // 左移缓存
    memmove(state->cache_pitch,
            state->cache_pitch + shift,
            (1024 - shift) * sizeof(int64_t));
    memmove(state->cache_pitchf,
            state->cache_pitchf + shift,
            (1024 - shift) * sizeof(float));

    // 插入新值 (跳过前3帧和最后1帧)
    size_t insert_pos = 1024 - (new_length - 4);
    size_t insert_len = new_length - 4;
    memcpy(state->cache_pitch + insert_pos,
           new_pitch + 3,
           insert_len * sizeof(int64_t));
    memcpy(state->cache_pitchf + insert_pos,
           new_pitchf + 3,
           insert_len * sizeof(float));
}
```

### 特征 2x 上采样

```c
void rvc_interpolate_features_2x(
    const float* input,      // [frames, 768]
    size_t input_frames,
    float* output,           // [frames*2, 768]
    size_t feature_dim       // 768
) {
    for (size_t i = 0; i < input_frames; i++) {
        // 复制原始帧
        memcpy(output + (i * 2) * feature_dim,
               input + i * feature_dim,
               feature_dim * sizeof(float));

        // 插值帧
        if (i < input_frames - 1) {
            for (size_t d = 0; d < feature_dim; d++) {
                output[(i * 2 + 1) * feature_dim + d] =
                    (input[i * feature_dim + d] +
                     input[(i + 1) * feature_dim + d]) * 0.5f;
            }
        } else {
            memcpy(output + (i * 2 + 1) * feature_dim,
                   input + i * feature_dim,
                   feature_dim * sizeof(float));
        }
    }
}
```

## 推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| block_size | 16000 | 1 秒 @ 16kHz |
| audio_context_size | 4800 | 300ms 上下文 |
| crossfade_samples | 480 | 10ms @ 48kHz |
| pitch_cache_size | 1024 | ~10 秒历史 |

## 内存需求

| 缓冲区 | 大小 | 说明 |
|--------|------|------|
| cache_pitch | 8 KB | 1024 * int64 |
| cache_pitchf | 4 KB | 1024 * float |
| audio_context | 19 KB | 4800 * float |
| feature_context | 60 KB | 100 * 768 * float |
| 工作缓冲区 | ~2 MB | 临时分配 |
| **总计** | ~2.1 MB | 流式状态 |

## 待完成工作

### 1. ✅ 实现 `src/rvc_streaming.cpp` (基础框架已完成)

已实现以下函数：
- ✅ `rvc_stream_default_config()`
- ✅ `rvc_stream_state_create()`
- ✅ `rvc_stream_state_destroy()`
- ✅ `rvc_stream_state_reset()`
- ✅ `rvc_stream_process()` (占位实现)
- ✅ `rvc_stream_flush()` (占位实现)
- ✅ `rvc_interpolate_features_2x()`
- ✅ `rvc_f0_to_pitch()`
- ✅ `rvc_f0_to_pitch_batch()`

### 2. ✅ 修改 `CMakeLists.txt`

已添加 `src/rvc_streaming.cpp` 到源文件列表。

### 3. ✅ 创建测试程序

`tests/test_streaming.cpp` 包含：
- ✅ 默认配置测试
- ✅ 状态创建/销毁测试
- ✅ 状态重置测试
- ✅ F0 转 Pitch 测试
- ✅ 批量 F0 转 Pitch 测试
- ✅ 特征 2x 上采样测试
- ✅ 输出大小计算测试
- ✅ 单块处理测试
- ✅ 错误字符串测试
- ✅ 多块连续处理测试
- ✅ 刷新测试

### 4. ⏳ 集成实际的 ONNX 推理

待集成到 `rvc_stream_process()`:
- F0 实际提取 (调用 `f0_extractor.h`)
- ContentVec/HuBERT 特征提取 (调用 `onnx_inference.h`)
- 合成器推理 (调用 `onnx_inference.h`)

### 5. ⏳ 实时音频测试

创建使用 PortAudio 的实时音频测试程序。

## 注意事项

1. **第一块处理** - pitch 缓存初始化为 0，需要特殊处理
2. **F0 上下文** - F0 提取需要额外的音频上下文以保证边界准确
3. **内存管理** - 所有缓冲区在 `rvc_stream_state_create()` 中分配
4. **线程安全** - 当前设计不是线程安全的，每个流需要独立的状态

## 参考资料

- Python 实现: `infer/lib/rtrvc.py`
- 现有 C 实现: `RVC_C/src/rvc_onnx.cpp`
- ONNX 推理接口: `RVC_C/include/onnx_inference.h`
- F0 提取接口: `RVC_C/include/f0_extractor.h`

---

*文档创建时间: 2026-02-02*
*最后更新: 2026-02-07*
*当前状态: ✅ 流式处理实现完成，端到端测试通过*

## 测试结果

### 单元测试 (test_streaming)
- **状态**: 11/11 通过 ✅
- 默认配置、状态管理、F0转Pitch、特征上采样等全部通过

### 端到端测试 (test_streaming_e2e)
- **状态**: 通过 ✅
- 测试音频: 19.63 秒
- **SOLA 平滑已启用**: 帧间过渡更平滑

#### 最新测试结果 (SOLA 启用)
| 块大小 | RTF | 输出文件 |
|--------|-----|----------|
| 500ms | 0.70 | streaming_output_sola.wav |
| 300ms | 0.82 | streaming_output_sola_300ms.wav |

### 性能指标
| 块大小 | RTF | 实时性 |
|--------|-----|--------|
| 500ms | ~0.70 | ✅ 实时 |
| 300ms | ~0.82 | ✅ 实时 |

**注**: 使用 DIO 替代 Harvest 可进一步提升速度

## SOLA 算法实现

### 概述

SOLA (Synchronous Overlap-Add) 算法用于消除帧间音频不连续性（卡顿/咔哒声）。

### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| RVC_SOLA_OVERLAP_SIZE | 480 | 重叠窗口大小 (~10ms @ 48kHz) |
| RVC_SOLA_SEARCH_RANGE | 240 | 搜索范围 (~5ms @ 48kHz) |

### 算法流程

```
1. 保存前一块输出的尾部 (output_tail)

2. 处理下一块时:
   a. 计算归一化互相关，找到最佳对齐位置
      - 在当前块开头的搜索范围内
      - 找到与 output_tail 最相似的位置

   b. 应用线性交叉淡化
      - output[i] = prev_tail[i] * (1-t) + curr[i] * t
      - t 从 0 递增到 1

   c. 复制剩余部分

3. 保存当前块尾部用于下次处理
```

### 核心函数

```c
// 查找最佳重叠偏移 (归一化互相关)
static size_t sola_find_best_offset(
    const float* ref,      // 参考信号 (前一块尾部)
    size_t ref_len,
    const float* search,   // 搜索信号 (当前块开头)
    size_t search_len
);

// 线性交叉淡化
static void sola_crossfade(
    float* output,
    const float* prev_tail,
    const float* curr_start,
    size_t fade_len
);
```

### 数据结构更新

```c
typedef struct RVCStreamState {
    // ... 原有字段 ...

    // SOLA 输出缓冲 (用于帧间平滑)
    float* output_tail;             /**< 前一块输出的尾部 (用于 SOLA) */
    size_t output_tail_size;        /**< 输出尾部大小 */
    size_t output_tail_capacity;    /**< 输出尾部容量 */
} RVCStreamState;
```

