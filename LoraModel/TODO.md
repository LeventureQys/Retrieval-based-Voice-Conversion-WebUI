# RVC-LoRA 待办事项

> 最后更新: 2026-01-28

---

## 🔴 紧急 - 需要立即修复

### PyTorch 2.6 兼容性问题

**问题**: PyTorch 2.6 将 `torch.load` 的 `weights_only` 默认值改为 `True`，导致 fairseq 加载 HuBERT 模型失败。

**待修复文件**:
- [ ] `inference/infer_lora.py` - `_load_hubert()` 方法

**已修复文件**:
- [x] `preprocessing/feature_extractor.py`
- [x] `scripts/infer_lora_e2e.py`

**修复方法**: 在调用 fairseq 加载模型前，临时 patch `torch.load`:
```python
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
try:
    # fairseq 加载代码
finally:
    torch.load = original_load
```

---

## 🟡 高优先级 - 端到端测试

修复兼容性问题后，需要完成以下测试：

- [ ] 使用 `base_voice` 数据训练 LoRA (10-20 epochs 快速验证)
- [ ] 使用 `test_voice` 数据测试推理
- [ ] 评估转换质量指标 (MCD, F0 Correlation, Spectral Convergence)
- [ ] 验证 LoRA 权重文件大小
- [ ] 记录训练时间

**测试命令**:
```bash
# 从 LoraModel 目录运行
python scripts/test_e2e.py --epochs 10 --batch_size 2
```

---

## 🟢 中优先级 - 功能完善

- [ ] 添加训练进度可视化 (TensorBoard 已支持)
- [ ] 添加早停机制
- [ ] 支持多 GPU 训练
- [ ] 添加数据增强选项

---

## 🔵 低优先级 - 文档和优化

- [ ] 完善 API 文档
- [ ] 添加更多使用示例
- [ ] 优化内存使用
- [ ] 性能基准测试

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
