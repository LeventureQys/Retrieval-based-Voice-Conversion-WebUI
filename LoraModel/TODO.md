# RVC-LoRA 待办事项

> 最后更新: 2026-01-29
>
> **项目状态: ⏸️ 暂停开发**

---

## ⏸️ 项目暂停

### 暂停原因

经过深入实验和分析，决定暂停 RVC-LoRA 项目开发：

| 问题 | 详情 |
|------|------|
| **RVC 模型太小** | 仅 ~90MB，全量训练本身就很快，LoRA 省时优势不明显 |
| **训练已经很轻量** | RVC 原生训练显存需求仅 4-8GB，LoRA 省显存优势不明显 |
| **Loss 收敛有限** | Mel Loss 从 0.50 下降到 0.42 后趋于平稳（下降 16%），继续训练改善有限 |
| **社区无采用** | RVC 社区几乎没有使用 LoRA 的案例，说明实际需求不强 |

### 实验记录

**训练配置**:
- 数据: 736 个音频片段（约 30 分钟语音，降噪后）
- 底模: f0G40k.pth (v2)
- 损失函数: Mel Spectrogram L1 Loss
- 学习率调度: Warmup + Cosine Annealing

**rank=8 实验结果**:
| Epoch | Loss | 变化 |
|-------|------|------|
| 0 | 0.5039 | - |
| 5 | 0.4390 | -12.9% |
| 10 | 0.4270 | -15.3% |
| 14 | 0.4236 | -15.9% |

**结论**: Loss 在 0.42 左右趋于平稳，继续训练或增大 rank 改善有限。

### LoRA 适用场景分析

LoRA 更适合以下场景：

| 适合 | 不适合 |
|------|--------|
| 大模型 (>1GB) | 小模型 (<100MB) ← RVC |
| 显存受限 | 显存充足 |
| 需要多个变体快速切换 | 只需 1-2 个模型 |
| 训练数据极少 | 训练数据充足 |

### 建议

**如果目标是训练 RVC 声音模型，直接使用 RVC 原生训练更简单高效。**

---

## 已完成 ✅

### 核心功能
- [x] LoRA 层实现 (Linear, Conv1d, ConvTranspose1d)
- [x] LoRA 注入/提取/合并工具
- [x] GeneratorLoRA 模型
- [x] SynthesizerLoRA 包装器
- [x] 训练流程 (Trainer, DataLoader, Losses)
- [x] 推理流程 (Inference, ModelLoader)

### 端到端管道
- [x] 音频预处理 (切片、归一化、重采样)
- [x] 特征提取 (HuBERT, F0, Mel)
- [x] 端到端训练脚本 (`train_lora_e2e.py`)
- [x] 端到端推理脚本 (`infer_lora_e2e.py`)
- [x] 端到端测试脚本 (`test_e2e.py`)

### 测试
- [x] 单元测试 (40/40 通过)
- [x] 模型集成测试

---

## 快速参考

### 项目结构
```
LoraModel/
├── preprocessing/     # 预处理 (音频处理、特征提取)
├── lora/             # LoRA 核心
├── models/           # 模型定义
├── training/         # 训练代码
├── inference/        # 推理代码
├── scripts/          # 端到端脚本 ⭐
└── tests/            # 测试代码
```

### 关键文件
| 文件 | 用途 |
|-----|------|
| `scripts/train_lora_e2e.py` | 端到端训练入口 |
| `scripts/infer_lora_e2e.py` | 端到端推理入口 |
| `scripts/test_e2e.py` | 完整测试脚本 |
| `preprocessing/feature_extractor.py` | 特征提取 (HuBERT/F0/Mel) |
| `training/train_lora.py` | LoRA 训练器 |

### 测试数据
- 训练: `download/base_voice/` (6个文件, 135秒)
- 测试: `download/test_voice/` (2个文件, 30秒)
