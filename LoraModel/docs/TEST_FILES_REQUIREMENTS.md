# 测试文件需求说明

## 概述

本文档说明 RVC-LoRA 项目在训练和推理阶段需要的测试文件。

---

## 阶段 2: 模型集成测试

### 需要的文件

#### 1. 底模文件 (Base Model)
**位置**: `../assets/pretrained/` 或 `../assets/pretrained_v2/`

**必需文件**:
```
G40k.pth          # 生成器底模 (40kHz, 推荐)
或
G48k.pth          # 生成器底模 (48kHz, 高质量)
或
G32k.pth          # 生成器底模 (32kHz, 快速)
```

**文件说明**:
- 这是预训练的 RVC 生成器模型
- 用于初始化模型权重
- 大小约 55 MB
- 可以从 [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2) 下载

**测试用途**:
- 加载底模并注入 LoRA
- 验证前向传播
- 测试模型结构

---

## 阶段 3: 训练测试

### 需要的文件

#### 1. 训练数据
**位置**: `./test_data/training/`

**文件结构**:
```
test_data/
└── training/
    ├── audio/              # 音频文件
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    ├── features/           # 提取的特征 (可选)
    │   └── ...
    └── metadata.txt        # 元数据 (可选)
```

**音频要求**:
- 格式: WAV, 16-bit PCM
- 采样率: 40kHz (或与底模匹配)
- 时长: 每个文件 2-10 秒
- 数量: 至少 10 个文件用于测试
- 总时长: 建议 1-2 分钟 (仅用于测试)

**数据说明**:
- 这些是用于训练 LoRA 的目标说话人语音
- 测试时不需要大量数据
- 可以使用任何干净的人声录音

#### 2. 底模文件
**位置**: `../assets/pretrained_v2/`

**必需文件**:
```
G40k.pth          # 生成器底模
D40k.pth          # 判别器底模 (训练时需要)
```

**文件说明**:
- G40k.pth: 生成器，用于语音合成
- D40k.pth: 判别器，用于对抗训练
- 两者都需要用于完整的训练流程

#### 3. HuBERT 模型
**位置**: `../assets/hubert/`

**必需文件**:
```
hubert_base.pt    # HuBERT 特征提取模型
```

**文件说明**:
- 用于提取语音特征
- 大小约 189 MB
- 已经下载完成

#### 4. 配置文件
**位置**: `./configs/`

**文件内容**:
```json
{
  "train": {
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 100,
    "save_interval": 10
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["ups", "resblocks"]
  },
  "model": {
    "sampling_rate": 40000,
    "hop_length": 400
  }
}
```

---

## 阶段 4: 推理测试

### 需要的文件

#### 1. 输入音频
**位置**: `./test_data/inference/input/`

**文件要求**:
```
input_001.wav     # 测试输入音频
input_002.wav
...
```

**音频要求**:
- 格式: WAV
- 采样率: 任意 (会自动重采样)
- 时长: 5-30 秒
- 内容: 任何人声录音

**用途**:
- 测试语音转换功能
- 验证 LoRA 推理效果

#### 2. 训练好的 LoRA 权重
**位置**: `./output/` 或 `./checkpoints/`

**文件**:
```
lora_weights.pth  # LoRA 权重文件
或
checkpoint_epoch_100.pth  # 完整检查点
```

**文件说明**:
- 这是训练阶段产生的输出
- 包含 LoRA 参数
- 大小约 400 KB - 1 MB

#### 3. 底模文件
**位置**: `../assets/pretrained_v2/`

**必需文件**:
```
G40k.pth          # 生成器底模
```

#### 4. 索引文件 (可选)
**位置**: `./test_data/inference/`

**文件**:
```
speaker_index.index  # FAISS 索引文件
```

**文件说明**:
- 用于特征检索
- 可以提高音色相似度
- 非必需，可以设置 index_rate=0 跳过

---

## 最小测试集

### 快速测试所需的最小文件集

```
必需文件:
1. ../assets/pretrained_v2/G40k.pth        (底模)
2. ../assets/hubert/hubert_base.pt         (已有)
3. ./test_data/training/audio/*.wav        (10个音频文件)
4. ./test_data/inference/input/*.wav       (2-3个测试音频)

可选文件:
5. ../assets/pretrained_v2/D40k.pth        (训练时需要)
6. ./test_data/inference/*.index           (索引文件)
```

---

## 文件准备建议

### 方案 1: 使用真实数据
如果你有目标说话人的录音:
```bash
# 1. 准备训练数据
mkdir -p test_data/training/audio
# 复制 10-20 个音频文件到 test_data/training/audio/

# 2. 准备测试数据
mkdir -p test_data/inference/input
# 复制 2-3 个测试音频到 test_data/inference/input/
```

### 方案 2: 使用合成数据 (仅用于功能测试)
```python
# 生成测试音频
import numpy as np
import soundfile as sf

for i in range(10):
    # 生成 3 秒的随机音频
    audio = np.random.randn(40000 * 3) * 0.1
    sf.write(f'test_data/training/audio/sample_{i:03d}.wav',
             audio, 40000)
```

### 方案 3: 使用现有 RVC 数据
如果你已经有 RVC 训练数据:
```bash
# 使用现有的数据集
ln -s /path/to/existing/dataset test_data/training
```

---

## 测试流程

### 阶段 2 测试 (模型集成)
```bash
# 只需要底模
需要: G40k.pth
测试: 模型加载、LoRA 注入、前向传播
```

### 阶段 3 测试 (训练)
```bash
# 需要底模 + 训练数据
需要: G40k.pth, D40k.pth, training/audio/*.wav
测试: 完整训练流程
```

### 阶段 4 测试 (推理)
```bash
# 需要底模 + LoRA 权重 + 输入音频
需要: G40k.pth, lora_weights.pth, input/*.wav
测试: 语音转换推理
```

---

## 文件大小参考

```
底模文件:
- G40k.pth: ~55 MB
- D40k.pth: ~55 MB

特征提取:
- hubert_base.pt: ~189 MB (已有)

训练数据:
- 10个音频文件: ~10-50 MB (取决于时长)

LoRA 权重:
- lora_weights.pth: ~400 KB - 1 MB

测试音频:
- 每个文件: ~1-5 MB
```

---

## 下载链接

### 底模下载
```
Hugging Face:
https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2

需要下载:
- G40k.pth
- D40k.pth (训练时需要)
```

### 测试数据
```
可以使用:
1. 自己的录音
2. 公开数据集 (如 VCTK)
3. 合成数据 (仅功能测试)
```

---

## 注意事项

1. **底模必需**: 所有阶段都需要 G40k.pth
2. **训练数据质量**: 影响最终效果，但测试时可以用少量数据
3. **采样率匹配**: 确保音频采样率与底模匹配 (40kHz)
4. **文件路径**: 确保文件路径正确，可以使用相对路径或绝对路径

---

## 当前状态

### 单元测试 (已完成)
```
✅ test_lora_core.py    - 6/6 通过
✅ test_model.py        - 7/7 通过
✅ test_training.py     - 11/11 通过
✅ test_inference.py    - 8/8 通过
总计: 32/32 通过
```

### 端到端测试 (需要准备文件)
```
已有文件:
✅ hubert_base.pt (189 MB)
✅ rmvpe.pt (181 MB)

需要准备:
⏸️ G40k.pth (底模) - 必需
⏸️ D40k.pth (训练时需要) - 可选
⏸️ 训练音频数据 (5-10个文件) - 训练测试需要
⏸️ 测试音频数据 (2-3个文件) - 推理测试需要
```

---

## 快速准备指南

### 最小测试集 (仅测试代码功能)

只需要准备以下文件即可进行基本的端到端测试：

1. **底模文件** (必需)
   - 下载 `G40k.pth` 到 `assets/pretrained_v2/`
   - 下载链接: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2

2. **训练数据** (5个样本即可)
   ```
   LoraModel/test_data/
   ├── sample0.spec.pt      # Mel频谱 [80, T]
   ├── sample0_phone.npy    # 音素特征 [T, 256]
   ├── sample0_pitch.npy    # 量化音高 [T]
   ├── sample0_pitchf.npy   # 连续音高 [T]
   └── ... (重复4次)
   ```

### 生成模拟测试数据

如果暂时没有真实数据，可以运行以下脚本生成模拟数据：

```python
import torch
import numpy as np
import os

os.makedirs('test_data', exist_ok=True)

for i in range(5):
    T = 100  # 时间步

    # Mel 频谱
    spec = torch.randn(80, T)
    torch.save(spec, f'test_data/sample{i}.spec.pt')

    # 音素特征 (v1: 256维, v2: 768维)
    phone = np.random.randn(T, 256).astype(np.float32)
    np.save(f'test_data/sample{i}_phone.npy', phone)

    # 量化音高
    pitch = np.random.randint(0, 256, T).astype(np.int64)
    np.save(f'test_data/sample{i}_pitch.npy', pitch)

    # 连续音高
    pitchf = np.random.randn(T).astype(np.float32) * 100 + 200
    np.save(f'test_data/sample{i}_pitchf.npy', pitchf)

print("模拟测试数据已生成到 test_data/ 目录")
```

---

## 联系方式

如果你准备好了测试文件，请告诉我文件的位置，我会继续进行测试和验证。

**准备好后，请提供**:
1. 底模文件路径
2. 训练数据路径 (如果要测试训练)
3. 测试音频路径 (如果要测试推理)

---

**文档版本**: 1.0
**更新日期**: 2026-01-28
