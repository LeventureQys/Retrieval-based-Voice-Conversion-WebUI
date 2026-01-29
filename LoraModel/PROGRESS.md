# RVC-LoRA 项目进度追踪

## 当前状态
- **当前阶段**: ⏸️ 项目暂停
- **开始日期**: 2026-01-28
- **最后更新**: 2026-01-29

---

## ⏸️ 项目暂停说明

### 暂停原因

经过深入实验和分析，决定暂停 RVC-LoRA 项目开发。

| 问题 | 详情 |
|------|------|
| **RVC 模型太小** | 仅 ~90MB，全量训练本身就很快（几分钟到几十分钟） |
| **训练已经很轻量** | RVC 原生训练显存需求仅 4-8GB |
| **Loss 收敛有限** | Mel Loss 从 0.50 下降到 0.42 后趋于平稳（下降 16%） |
| **社区无采用** | RVC 社区几乎没有使用 LoRA 的案例 |

### 最终实验数据

**训练配置**:
- 数据: 736 个音频片段（约 30 分钟语音，降噪后）
- 底模: f0G40k.pth (v2)
- 损失函数: Mel Spectrogram L1 Loss
- 学习率调度: Warmup + Cosine Annealing

**rank=8 训练结果**:
| Epoch | Mel Loss | 变化 |
|-------|----------|------|
| 0 | 0.5039 | - |
| 5 | 0.4390 | -12.9% |
| 10 | 0.4270 | -15.3% |
| 14 | 0.4236 | -15.9% |

**结论**: Loss 在 0.42 左右趋于平稳，LoRA 的表达能力有限。

### LoRA 适用场景

LoRA 更适合以下场景，而非 RVC：

| 适合 | 不适合 |
|------|--------|
| 大模型 (>1GB) 如 SD, LLaMA | 小模型 (<100MB) ← RVC |
| 显存受限环境 | 显存充足 |
| 需要多个变体快速切换 | 只需 1-2 个模型 |

### 建议

**如果目标是训练 RVC 声音模型，直接使用 RVC 原生训练更简单高效。**

---

## 阶段进度

### ✅ 项目准备阶段
**状态**: 已完成
**完成日期**: 2026-01-28

- [x] 创建项目目录结构
- [x] 编写项目大纲 (PROJECT_OUTLINE.md)
- [x] 编写进度追踪文档 (PROGRESS.md)
- [x] 规划开发路线图

---

### ✅ 阶段 1: LoRA 核心实现
**状态**: 已完成
**完成日期**: 2026-01-28

#### 交付物
- ✅ `lora/lora_config.py` - LoRA 配置类
- ✅ `lora/lora_layer.py` - LoRA 层实现 (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)
- ✅ `lora/lora_utils.py` - 工具函数 (注入、提取、合并、保存/加载)
- ✅ `tests/test_lora_core.py` - 单元测试 (6/6 通过)

---

### ✅ 阶段 2: 模型集成
**状态**: 已完成
**完成日期**: 2026-01-28

#### 交付物
- ✅ `models/resblock.py` - ResBlock 实现
- ✅ `models/generator_lora.py` - GeneratorLoRA 类
- ✅ `models/synthesizer_lora.py` - SynthesizerLoRA 包装器
- ✅ `tests/test_model.py` - 单元测试 (7/7 通过)

#### LoRA 参数统计
```
Total parameters: 15,457,280
LoRA parameters: 342,912 (2.22%)
```

---

### ✅ 阶段 3: 训练流程
**状态**: 已完成
**完成日期**: 2026-01-28

#### 交付物
- ✅ `training/losses.py` - 损失函数
- ✅ `training/data_loader.py` - 数据加载器
- ✅ `training/train_lora.py` - 训练脚本
- ✅ `tests/test_training.py` - 单元测试 (11/11 通过)

---

### ✅ 阶段 4: 推理实现
**状态**: 已完成
**完成日期**: 2026-01-28

#### 交付物
- ✅ `inference/model_loader.py` - 模型加载器
- ✅ `inference/infer_lora.py` - 推理类
- ✅ `scripts/merge_lora.py` - LoRA 合并脚本
- ✅ `tests/test_inference.py` - 单元测试 (8/8 通过)

---

### ✅ 阶段 5: 端到端管道 (新增)
**状态**: 已完成
**完成日期**: 2026-01-28

#### 新增功能
为解决项目无法直接用于实际训练和推理的问题，新增了完整的端到端管道：

#### 交付物
- ✅ `preprocessing/__init__.py` - 预处理模块
- ✅ `preprocessing/audio_processor.py` - 音频处理 (加载、切片、归一化)
- ✅ `preprocessing/feature_extractor.py` - 特征提取 (HuBERT, F0, Mel)
- ✅ `preprocessing/pipeline.py` - 完整预处理管道
- ✅ `scripts/train_lora_e2e.py` - 端到端训练脚本
- ✅ `scripts/infer_lora_e2e.py` - 端到端推理脚本
- ✅ `scripts/test_e2e.py` - 端到端测试脚本
- ✅ `training/data_loader.py` - 更新支持 PreprocessedDataset

#### 数据流
```
训练流程:
原始音频 → [audio_processor] → 切片/归一化 → [feature_extractor] →
HuBERT/F0/Mel特征 → [train_lora_e2e] → LoRA权重

推理流程:
源音频 → [infer_lora_e2e] → HuBERT/F0提取 → 模型推理 → 转换后的音频
```

---

### ✅ 阶段 6: 完整测试和优化
**状态**: 已完成
**完成日期**: 2026-01-29

#### 任务清单
- [x] 修复 PyTorch 2.6 兼容性问题
- [x] 使用真实数据进行端到端测试
- [x] 性能基准测试
- [x] 质量对比测试
- [x] 存储空间验证
- [x] 训练时间验证

#### 端到端测试结果 (2026-01-29)
```
训练配置:
- 训练数据: 6个音频文件, 47个片段
- Epochs: 10
- Batch size: 2
- LoRA rank: 8, alpha: 16
- 设备: CUDA (GPU)

训练结果:
- 训练时间: 25.1秒
- 最终 Loss: 0.1069
- LoRA 参数: 384,768 (1.05%)

推理结果:
- 转换文件: 2个
- 推理时间: ~0.5-0.7秒/文件

质量指标:
- F0 Correlation: 0.966 (7.wav) - 优秀
- Spectral Convergence: 0.341 - 良好
- MCD: 需要进一步优化

存储空间:
- LoRA 权重文件: 1.5 MB (lora_final.pth)
- 完整 checkpoint: 4.8 MB (含优化器状态)
- 对比原始模型: ~140 MB → 节省 99%
```

#### 测试数据
```
训练数据 (base_voice): 6个文件, 总时长 134.68秒
- 1.wav: 18.62s, 48kHz, stereo
- 2.wav: 21.06s, 48kHz, stereo
- 3.wav: 16.96s, 48kHz, stereo
- 4.wav: 17.88s, 48kHz, stereo
- 5.wav: 17.86s, 48kHz, stereo
- 6.wav: 42.30s, 48kHz, stereo

测试数据 (test_voice): 2个文件
- 7.wav: 19.63s, 48kHz, stereo
- enrollment_000001.wav: 10.40s, 16kHz, mono
```

---

## 测试汇总

### 单元测试结果
| 测试文件 | 通过/总数 | 状态 |
|---------|----------|------|
| test_lora_core.py | 6/6 | ✅ |
| test_model.py | 7/7 | ✅ |
| test_training.py | 11/11 | ✅ |
| test_inference.py | 8/8 | ✅ |
| test_e2e.py | 8/8 | ✅ |
| **总计** | **40/40** | ✅ |

### 端到端测试结果
| 测试项 | 状态 | 备注 |
|-------|------|------|
| 音频预处理 | ✅ | 47个片段生成成功 |
| HuBERT 特征提取 | ✅ | PyTorch 2.6 兼容性已修复 |
| F0 特征提取 | ✅ | RMVPE 模型正常工作 |
| LoRA 训练 | ✅ | 10 epochs, loss 0.1069 |
| 推理转换 | ✅ | 2个文件成功转换 |
| 质量评估 | ✅ | F0 Correlation 0.966 |

---

## 项目结构

```
LoraModel/
├── lora/                   # LoRA 核心实现 ✅
│   ├── __init__.py
│   ├── lora_config.py      # 配置类
│   ├── lora_layer.py       # LoRA 层
│   └── lora_utils.py       # 工具函数
│
├── models/                 # 模型定义 ✅
│   ├── __init__.py
│   ├── resblock.py         # ResBlock
│   ├── generator_lora.py   # GeneratorLoRA
│   └── synthesizer_lora.py # SynthesizerLoRA
│
├── preprocessing/          # 预处理模块 ✅ (新增)
│   ├── __init__.py
│   ├── audio_processor.py  # 音频处理
│   ├── feature_extractor.py # 特征提取 (需修复)
│   └── pipeline.py         # 完整管道
│
├── training/               # 训练代码 ✅
│   ├── __init__.py
│   ├── losses.py           # 损失函数
│   ├── data_loader.py      # 数据加载 (已更新)
│   └── train_lora.py       # 训练脚本
│
├── inference/              # 推理代码 ✅
│   ├── __init__.py
│   ├── model_loader.py     # 模型加载
│   └── infer_lora.py       # 推理脚本 (需修复)
│
├── scripts/                # 端到端脚本 ✅ (新增)
│   ├── train_lora_e2e.py   # 端到端训练
│   ├── infer_lora_e2e.py   # 端到端推理 (已修复)
│   ├── test_e2e.py         # 端到端测试
│   └── merge_lora.py       # LoRA 合并
│
├── tests/                  # 测试代码 ✅
│   ├── test_lora_core.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_e2e.py
│
├── download/               # 预训练模型
│   ├── pretrained_v2/      # RVC 预训练模型 ✅
│   ├── hubert_base.pt      # HuBERT 模型 ✅
│   ├── base_voice/         # 训练音频 ✅
│   └── test_voice/         # 测试音频 ✅
│
├── docs/                   # 文档
├── examples/               # 示例代码
│
├── README.md               # 项目说明 ✅ (已更新)
├── PROGRESS.md             # 进度追踪 (本文件)
├── PROJECT_OUTLINE.md      # 项目大纲
└── requirements.txt        # 依赖列表
```

---

## 使用方法

### 端到端训练 (推荐)
```bash
python scripts/train_lora_e2e.py \
    --input_dir ./download/base_voice \
    --output_dir ./output \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --epochs 100
```

### 端到端推理
```bash
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth
```

---

## 下一步计划

1. **修复 PyTorch 2.6 兼容性问题**
   - 修复 `inference/infer_lora.py` 中的 HuBERT 加载
   - 验证所有模块的兼容性

2. **完成端到端测试**
   - 使用 base_voice 数据训练 LoRA
   - 使用 test_voice 数据测试推理
   - 评估转换质量 (MCD, F0 Correlation, Spectral Convergence)

3. **性能优化**
   - 优化特征提取速度
   - 优化训练内存使用

---

## 质量评估指标

测试脚本 (`scripts/test_e2e.py`) 实现了以下评估指标：

| 指标 | 说明 | 优秀 | 良好 | 可接受 |
|-----|------|------|------|--------|
| MCD (Mel Cepstral Distortion) | 频谱相似度，越低越好 | < 4.0 dB | 4.0-6.0 dB | 6.0-8.0 dB |
| F0 Correlation | 音高跟踪准确度，越高越好 | > 0.9 | 0.7-0.9 | < 0.7 |
| Spectral Convergence | 频谱收敛度，越低越好 | < 0.2 | 0.2-0.5 | > 0.5 |

---

**最后更新**: 2026-01-28
**下次更新**: 修复兼容性问题并完成端到端测试后
