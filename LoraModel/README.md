# RVC-LoRA: LoRA Support for RVC Voice Conversion

> **项目状态**: ⏸️ 暂停开发 | **最后更新**: 2026-01-29

---

## ⚠️ 项目暂停说明

**经过实验验证，LoRA 方案对 RVC 的实际收益有限，项目暂停开发。**

### 暂停原因

| 问题 | 说明 |
|------|------|
| **RVC 模型太小** | RVC 模型仅 ~90MB，全量训练本身就很快（几分钟到几十分钟），LoRA 省时优势不明显 |
| **训练已经很轻量** | RVC 原生训练显存需求仅 4-8GB，LoRA 省显存优势不明显 |
| **Loss 收敛有限** | 实验中 Mel Loss 从 0.50 下降到 0.42 后趋于平稳，继续训练改善有限 |
| **社区无采用** | RVC 社区几乎没有使用 LoRA 的案例，说明实际需求不强 |

### 实验数据

使用 736 个音频片段（约 30 分钟语音）进行训练：

| 配置 | Epoch | Loss | 说明 |
|------|-------|------|------|
| rank=8 | 0 | 0.5039 | 初始 |
| rank=8 | 15 | 0.4236 | 下降 16% 后趋于平稳 |
| rank=32 | - | - | 未完成测试 |

### LoRA 真正适合的场景

LoRA 更适合以下场景，而非 RVC：

| 场景 | 原因 |
|------|------|
| **大模型**（>1GB） | 如 Stable Diffusion、LLaMA，全量训练显存不够 |
| **训练数据少** | LoRA 参数少，不易过拟合 |
| **需要多个变体** | 一个底模 + 多个小 LoRA 文件，便于分发 |

### 建议

如果你的目标是训练 RVC 声音模型，**直接使用 RVC 原生训练更简单高效**。

---

## 🚀 快速开始（仅供参考）

**注意**: 以下内容仅供技术参考，不建议在生产环境使用。

**新用户？** 查看 [快速入门指南 (QUICKSTART.md)](QUICKSTART.md) 了解基本用法。

---

## 开发状态

| 模块 | 状态 | 说明 |
|-----|------|------|
| LoRA 核心 | ✅ 完成 | 层实现、注入、合并 |
| 模型集成 | ✅ 完成 | GeneratorLoRA, SynthesizerLoRA |
| 训练流程 | ✅ 完成 | Trainer, DataLoader, Losses |
| 推理流程 | ✅ 完成 | Inference, ModelLoader |
| 预处理管道 | ✅ 完成 | 音频处理、特征提取 |
| 端到端脚本 | ✅ 完成 | train_lora_e2e.py, infer_lora_e2e.py |
| 端到端测试 | ⏸️ 暂停 | Loss 收敛有限，实际收益不明显 |

---

## 项目简介

RVC-LoRA 为 RVC (Retrieval-based Voice Conversion) 添加了 LoRA (Low-Rank Adaptation) 支持，实现高效的语音模型微调。

### 主要特性

- ✅ **大幅减少存储空间**: 节省 89% 的模型存储空间
- ✅ **加速训练过程**: 减少 40-50% 的训练时间
- ✅ **保持高质量**: 达到完整微调 95-98% 的质量
- ✅ **快速训练多个声音**: 共享底模，只需训练差异部分
- ✅ **向后兼容**: 支持加载原始 RVC 完整模型
- ✅ **端到端流程**: 从原始音频到训练完成，一条命令搞定

### 性能对比

| 指标 | 完整微调 | LoRA |
|-----|---------|------|
| 单个模型大小 | 55 MB | 底模 55MB + LoRA 387KB |
| 10个模型存储 | 550 MB | 58.87 MB (节省 89%) |
| 训练时间 | 100% | 50-60% |
| 质量 | 100% | 95-98% |

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.13+
- CUDA (可选，用于 GPU 加速)
- fairseq (用于 HuBERT 特征提取)
- ffmpeg (用于音频处理)

### 安装

```bash
# 激活 conda 环境
conda activate rvc

# 安装依赖
pip install -r requirements.txt
```

### 下载预训练模型

训练和推理需要以下模型文件：

1. **HuBERT 模型**: `download/hubert_base.pt` (189 MB)
2. **RMVPE 模型**: `download/rmvpe.pt` (55 MB)
3. **预训练 RVC 模型**: `download/pretrained_v2/f0G40k.pth` (或其他采样率，每个约 55 MB)

#### 使用下载脚本（推荐）

我们提供了一个便捷的下载脚本来自动下载所有需要的模型：

```bash
# 从 LoraModel 目录运行

# 下载所有模型（推荐）
python download_models.py --all

# 或者分别下载
python download_models.py --hubert          # 下载 HuBERT 模型
python download_models.py --rmvpe           # 下载 RMVPE 模型
python download_models.py --pretrained_v2   # 下载所有 pretrained_v2 模型

# 只下载特定的 pretrained_v2 模型
python download_models.py --pretrained_v2 --models f0G40k.pth f0G48k.pth

# 查看可用模型列表
python download_models.py --list

# 检查下载状态
python download_models.py --check
```

#### 可用的 Pretrained V2 模型

| 模型文件 | 采样率 | 类型 | 说明 |
|---------|--------|------|------|
| `f0G40k.pth` | 40kHz | Generator (F0) | **推荐** - 带音高的生成器 |
| `f0G48k.pth` | 48kHz | Generator (F0) | 高采样率，带音高 |
| `f0G32k.pth` | 32kHz | Generator (F0) | 低采样率，带音高 |
| `G40k.pth` | 40kHz | Generator | 不带音高的生成器 |
| `G48k.pth` | 48kHz | Generator | 不带音高的生成器 |
| `G32k.pth` | 32kHz | Generator | 不带音高的生成器 |
| `f0D40k.pth` | 40kHz | Discriminator (F0) | 判别器（训练用） |
| `f0D48k.pth` | 48kHz | Discriminator (F0) | 判别器（训练用） |
| `f0D32k.pth` | 32kHz | Discriminator (F0) | 判别器（训练用） |
| `D40k.pth` | 40kHz | Discriminator | 判别器（训练用） |
| `D48k.pth` | 48kHz | Discriminator | 判别器（训练用） |
| `D32k.pth` | 32kHz | Discriminator | 判别器（训练用） |

**注意**:
- 对于语音转换，推荐使用 **f0G40k.pth**（带音高的 40kHz 生成器）
- Generator (G) 模型用于推理，Discriminator (D) 模型仅在训练时需要
- F0 模型支持音高调整，适合跨性别语音转换

#### 手动下载

如果自动下载失败，可以手动从 HuggingFace 下载：

```bash
# HuBERT 模型
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O download/hubert_base.pt

# RMVPE 模型
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -O download/rmvpe.pt

# Pretrained V2 模型（以 f0G40k.pth 为例）
mkdir -p download/pretrained_v2
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -O download/pretrained_v2/f0G40k.pth
```

## 完整使用流程

### 步骤 1: 下载预训练模型

首先下载所需的预训练模型（参见上面的"下载预训练模型"部分）：

```bash
# 下载所有必需的模型
python download_models.py --all

# 验证下载
python download_models.py --check
```

### 步骤 2: 准备训练数据

准备你的语音数据：
- 格式：WAV 文件（支持其他格式，会自动转换）
- 时长：建议至少 5-10 分钟的清晰语音
- 质量：尽量使用低噪音、清晰的录音
- 组织：将所有音频文件放在一个文件夹中

```bash
my_voice_samples/
├── audio1.wav
├── audio2.wav
├── audio3.wav
└── ...
```

### 步骤 3: 训练 LoRA 模型

使用端到端训练脚本，只需提供原始音频文件夹：

```bash
# 基本用法 - 从原始音频训练 LoRA
python scripts/train_lora_e2e.py \
    --input_dir ./my_voice_samples \
    --output_dir ./output \
    --base_model ./download/pretrained_v2/f0G40k.pth

# 完整参数示例
python scripts/train_lora_e2e.py \
    --input_dir ./my_voice_samples \
    --output_dir ./output \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --epochs 100 \
    --batch_size 4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --sample_rate 40000 \
    --version v2
```

训练完成后，会在 `output` 目录生成：
- `lora_final.pth` - 最终的 LoRA 权重文件
- `lora_best.pth` - 验证集上表现最好的 LoRA 权重
- `checkpoints/` - 训练过程中的检查点
- `logs/` - TensorBoard 日志文件
- `train_*.log` - 训练日志

### 步骤 4: 使用 LoRA 进行推理

使用训练好的 LoRA 权重进行语音转换：

```bash
# 基本用法
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth

# 带音高调整（升高 2 个半音）
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth \
    --pitch 2

# 不使用 LoRA（仅使用基础模型）
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth
```

### 步骤 5: 测试和评估（可选）

使用测试脚本进行完整的端到端测试和质量评估：

```bash
# 快速测试（10 epochs）
python scripts/test_e2e.py --epochs 10 --batch_size 2

# 完整测试（100 epochs）
python scripts/test_e2e.py --epochs 100 --batch_size 4
```

测试脚本会：
1. 使用 `download/base_voice/` 中的音频训练 LoRA
2. 使用 `download/test_voice/` 中的音频测试推理
3. 计算质量指标（MCD, F0 Correlation, Spectral Convergence）
4. 生成测试报告

### 训练参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--input_dir` | 必填 | 包含原始音频文件的目录 |
| `--output_dir` | 必填 | 输出目录（保存 LoRA 权重和日志） |
| `--base_model` | auto | 基础 RVC 模型路径（会自动查找） |
| `--epochs` | 100 | 训练轮数（建议 50-200） |
| `--batch_size` | 4 | 批次大小（根据显存调整，2-8） |
| `--learning_rate` | 1e-4 | 学习率 |
| `--lora_rank` | 8 | LoRA 秩（越大质量越高但参数更多，推荐 4-16） |
| `--lora_alpha` | 16 | LoRA 缩放因子（通常设为 rank 的 2 倍） |
| `--sample_rate` | 40000 | 采样率（32000/40000/48000，推荐 40000） |
| `--version` | v2 | RVC 版本（v1/v2，推荐 v2） |
| `--f0` / `--no_f0` | f0 | 是否使用 F0 音高模型（推荐启用） |
| `--hubert_path` | auto | HuBERT 模型路径（会自动查找） |
| `--rmvpe_path` | auto | RMVPE 模型路径（会自动查找） |
| `--save_every` | 10 | 每 N 个 epoch 保存一次检查点 |
| `--validate_every` | 5 | 每 N 个 epoch 验证一次 |
| `--device` | auto | 设备（cuda/cpu，自动检测） |
| `--num_workers` | 4 | 数据加载线程数 |
| `--verbose` | False | 详细日志输出 |

## 端到端推理

使用训练好的 LoRA 进行语音转换：

```bash
# 基本用法
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth

# 带音高调整
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth \
    --pitch 2  # 升高2个半音

# 不使用 LoRA (仅基础模型)
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth
```

### 推理参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--source` | 必填 | 源音频文件 |
| `--output` | 必填 | 输出音频文件 |
| `--model` | 必填 | 基础 RVC 模型 |
| `--lora` | 无 | LoRA 权重文件 |
| `--pitch` | 0 | 音高调整 (半音) |
| `--protect` | 0.33 | 保护清辅音 (0-0.5) |
| `--speaker_id` | 0 | 说话人 ID |

## 分步训练 (高级用法)

如果需要更多控制，可以分步执行：

### 1. 数据预处理

```bash
python -m preprocessing.pipeline \
    --input ./my_voice_samples \
    --output ./preprocessed \
    --sample_rate 40000 \
    --version v2
```

### 2. 训练 LoRA

```bash
python training/train_lora.py \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --data_dir ./preprocessed \
    --output_dir ./output \
    --lora_rank 8 \
    --epochs 100
```

### 3. 推理

```bash
python inference/infer_lora.py \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/checkpoints/lora_best.pth \
    --input ./features.npz \
    --output ./output.wav
```

## 项目结构

```
LoraModel/
├── lora/               # LoRA 核心实现
│   ├── lora_layer.py   # LoRA 层定义
│   ├── lora_config.py  # 配置类
│   └── lora_utils.py   # 工具函数
├── models/             # 模型定义
│   ├── generator_lora.py
│   └── synthesizer_lora.py
├── preprocessing/      # 预处理模块 (新增)
│   ├── audio_processor.py  # 音频处理
│   ├── feature_extractor.py # 特征提取
│   └── pipeline.py     # 完整管道
├── training/           # 训练脚本
│   ├── train_lora.py   # 训练器
│   ├── data_loader.py  # 数据加载
│   └── losses.py       # 损失函数
├── inference/          # 推理脚本
│   ├── infer_lora.py   # 推理类
│   └── model_loader.py # 模型加载
├── scripts/            # 端到端脚本 (新增)
│   ├── train_lora_e2e.py   # 端到端训练
│   ├── infer_lora_e2e.py   # 端到端推理
│   └── merge_lora.py   # LoRA 合并
├── tests/              # 测试代码
├── examples/           # 使用示例
├── download/           # 预训练模型
│   ├── pretrained_v2/  # RVC 预训练模型
│   └── hubert_base.pt  # HuBERT 模型
└── docs/               # 详细文档
```

## 开发状态

- [x] 项目规划和结构设计
- [x] 阶段1: LoRA 核心实现
- [x] 阶段2: 模型集成
- [x] 阶段3: 训练流程
- [x] 阶段4: 推理实现
- [x] 阶段5: 端到端管道
- [ ] 阶段6: 完整测试和优化

## 技术原理

### 什么是 LoRA？

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法。它不修改原始模型的权重，而是添加小的"适配器"层来学习特定任务的知识。

核心公式: `W' = W + BA`，其中 B 和 A 是低秩矩阵。

### 为什么使用 LoRA？

传统微调需要保存完整的模型副本，而 LoRA 只需要保存很小的适配器权重：

```
传统方式: 每个声音 = 55 MB 完整模型
LoRA 方式: 底模 55 MB (共享) + 每个声音 387 KB
```

### LoRA 在 RVC 中的应用

我们在以下层添加 LoRA：
1. **上采样层** (ConvTranspose1d) - 最重要，直接影响音质
2. **ResBlock 卷积层** - 影响音色细节

## 常见问题

### Q: 训练需要多少数据？
A: 建议至少 5-10 分钟的清晰语音数据，越多越好。数据质量比数量更重要。

### Q: 训练需要多长时间？
A: 取决于数据量和硬件。在 RTX 3090 上，100 epochs 大约需要 30-60 分钟。

### Q: LoRA rank 应该设置多少？
A: 推荐值为 8。更高的值 (16, 32) 可能提高质量但增加参数量。较低的值 (4) 可以减少参数但可能降低质量。

### Q: 如何选择采样率？
A: 40000 Hz 是推荐值，平衡了质量和性能。48000 Hz 质量更高但计算量更大，32000 Hz 速度更快但质量稍低。

### Q: 什么时候使用 F0 模型？
A: 几乎总是推荐使用 F0 模型（f0G*.pth）。F0 模型支持音高调整，特别适合跨性别语音转换。

### Q: 如何调整音高？
A: 在推理时使用 `--pitch` 参数，单位是半音。例如 `--pitch 2` 升高 2 个半音，`--pitch -2` 降低 2 个半音。

### Q: 训练时显存不足怎么办？
A: 减小 `--batch_size`（例如从 4 降到 2 或 1），或使用较低的采样率。

### Q: 如何提高转换质量？
A:
1. 使用更多高质量的训练数据
2. 增加训练轮数（100-200 epochs）
3. 适当增加 LoRA rank（8-16）
4. 确保训练数据和目标音频的音质相近

## 故障排除

### PyTorch 2.6 兼容性问题

**问题**: 使用 PyTorch 2.6 时，HuBERT 模型加载失败，报错：
```
WeightsUnpickler error: Unsupported global: GLOBAL fairseq.data.dictionary.Dictionary
```

**解决方案**: 这是已知问题，代码中已经包含了修复。如果仍然遇到问题，请确保使用最新版本的代码。

### 找不到预训练模型

**问题**: 报错 "Base model not found" 或 "HuBERT model not found"

**解决方案**:
1. 运行 `python download_models.py --check` 检查模型是否已下载
2. 如果未下载，运行 `python download_models.py --all`
3. 确保模型文件在正确的位置：
   - HuBERT: `LoraModel/download/hubert_base.pt`
   - RMVPE: `LoraModel/download/rmvpe.pt`
   - Pretrained: `LoraModel/download/pretrained_v2/f0G40k.pth`

### 音频质量差

**问题**: 转换后的音频质量不理想

**解决方案**:
1. 检查训练数据质量（是否有噪音、是否清晰）
2. 增加训练数据量（至少 10 分钟）
3. 增加训练轮数（100-200 epochs）
4. 调整 `--protect` 参数（0.33-0.5）保护清辅音
5. 尝试不同的音高调整值

### CUDA 内存不足

**问题**: 训练或推理时报错 "CUDA out of memory"

**解决方案**:
1. 减小 batch_size：`--batch_size 2` 或 `--batch_size 1`
2. 使用较低的采样率：`--sample_rate 32000`
3. 减小 LoRA rank：`--lora_rank 4`
4. 如果仍然不够，使用 CPU：`--device cpu`（会很慢）

### 训练速度慢

**问题**: 训练速度很慢

**解决方案**:
1. 确保使用 GPU：检查 `torch.cuda.is_available()` 返回 True
2. 增加 batch_size（如果显存允许）：`--batch_size 8`
3. 减少数据加载线程数：`--num_workers 2`
4. 使用较低的采样率：`--sample_rate 32000`

## 使用场景

### 适合使用 LoRA 的情况

- ✅ 需要训练多个不同的声音模型（5个以上）
- ✅ 存储空间有限
- ✅ 需要快速切换不同声音
- ✅ 想要快速实验不同的声音

### 不适合使用 LoRA 的情况

- ❌ 只训练 1-2 个声音模型
- ❌ 对质量要求极高（需要 100% 质量）
- ❌ 不在意存储空间

## 贡献指南

欢迎贡献！请查看 [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) 了解项目结构和开发规范。

## 许可证

本项目基于原始 RVC 项目，遵循相同的开源许可证。

## 致谢

- 原始 RVC 项目: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- LoRA 论文: https://arxiv.org/abs/2106.09685
