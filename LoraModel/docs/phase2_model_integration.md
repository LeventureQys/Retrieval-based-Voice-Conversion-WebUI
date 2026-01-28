# 阶段 2: 模型集成文档

## 概述

本阶段将 LoRA (Low-Rank Adaptation) 集成到 RVC 的 Generator 和 Synthesizer 模型中，实现高效的模型微调。

## 完成的工作

### 1. ResBlock 实现 (`models/resblock.py`)

实现了两种 ResBlock 变体，与原始 RVC 保持兼容：

#### ResBlock1 (3路并行结构)
- 3 个并行卷积路径，膨胀率为 (1, 3, 5)
- 每个路径后跟一个膨胀率为 1 的卷积
- 使用 weight_norm 进行权重归一化
- LeakyReLU 激活函数 (斜率 0.1)

```python
class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        # convs1: 3个不同膨胀率的卷积
        # convs2: 3个膨胀率为1的卷积
```

#### ResBlock2 (2路并行结构)
- 2 个并行卷积路径，膨胀率为 (1, 3)
- 更轻量级的设计

```python
class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        # convs: 2个不同膨胀率的卷积
```

### 2. GeneratorLoRA 实现 (`models/generator_lora.py`)

带 LoRA 支持的 Generator 类，主要特性：

#### 架构
```
GeneratorLoRA
├── conv_pre: Conv1d (输入预处理)
├── ups: ModuleList (上采样层，支持 LoRA)
│   └── ConvTranspose1d with weight_norm
├── resblocks: ModuleList (残差块，支持 LoRA)
│   └── ResBlock1 或 ResBlock2
├── conv_post: Conv1d (输出后处理)
└── cond: Conv1d (条件输入，可选)
```

#### LoRA 注入
- 在上采样层 (ups) 注入 LoRA
- 在 ResBlock 的卷积层注入 LoRA
- 注入前自动移除 weight_norm

#### 使用示例
```python
from models import GeneratorLoRA
from lora import LoRAConfig

# 创建 LoRA 配置
lora_config = LoRAConfig(
    r=8,
    lora_alpha=16,
    target_modules=["ups", "resblocks"],
)

# 创建带 LoRA 的 Generator
generator = GeneratorLoRA(
    initial_channel=192,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 10, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[20, 20, 4, 4],
    gin_channels=256,
    lora_config=lora_config,
)
```

### 3. SynthesizerLoRA 实现 (`models/synthesizer_lora.py`)

Synthesizer 的 LoRA 包装器，支持所有 RVC Synthesizer 变体：

#### 支持的模型
- `SynthesizerTrnMs256NSFsid` (v1 with F0)
- `SynthesizerTrnMs768NSFsid` (v2 with F0)
- `SynthesizerTrnMs256NSFsid_nono` (v1 without F0)
- `SynthesizerTrnMs768NSFsid_nono` (v2 without F0)

#### 主要功能
- LoRA 注入到 Generator (dec) 组件
- 自动冻结非 LoRA 参数
- 支持 LoRA 权重的保存和加载
- 完整的 forward 和 infer 方法

#### 使用示例
```python
from models import SynthesizerLoRA, load_synthesizer_with_lora
from lora import LoRAConfig

# 方法 1: 从检查点加载
lora_config = LoRAConfig(r=8, lora_alpha=16, target_modules=["ups", "resblocks"])
model = load_synthesizer_with_lora(
    checkpoint_path="path/to/model.pth",
    lora_config=lora_config,
    device="cuda",
    version="v2",
    f0=True,
)

# 方法 2: 包装已有模型
from models import create_synthesizer_lora_from_pretrained
model = create_synthesizer_lora_from_pretrained(
    pretrained_model=existing_synthesizer,
    lora_config=lora_config,
)

# 获取 LoRA 参数用于优化器
lora_params = model.get_lora_parameters()
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

# 保存/加载 LoRA 权重
lora_state = model.get_lora_state_dict()
model.load_lora_weights(lora_state)
```

## Weight Norm 兼容性处理

### 问题
RVC 原始模型使用 `weight_norm` 进行权重归一化，这与 LoRA 不兼容。

### 解决方案
1. 在注入 LoRA 之前移除 weight_norm
2. 实现 `remove_weight_norm()` 方法递归移除所有层的 weight_norm
3. 实现 `__prepare_scriptable__()` 方法支持 TorchScript 编译

```python
def _inject_lora(self):
    # 先移除 weight_norm
    self.remove_weight_norm()
    # 然后注入 LoRA
    inject_lora(self, self.lora_config, target_modules=self.lora_config.target_modules)
```

## 参数冻结策略

### 默认策略
- 冻结所有基础模型参数
- 只训练 LoRA 参数 (lora_A, lora_B)

### 参数统计示例
```
Total parameters: 15,457,280
LoRA parameters: 342,912 (2.22%)
Base frozen: 15,114,368
LoRA trainable: 342,912
```

### 灵活控制
```python
# 解冻所有参数（完整微调）
model.unfreeze_all()

# 重新冻结基础模型
model.freeze_base_model()
```

## 测试结果

所有 7 个测试通过：

| 测试 | 状态 | 说明 |
|------|------|------|
| ResBlock1 | ✅ | 前向传播、mask 支持、weight_norm 移除 |
| ResBlock2 | ✅ | 前向传播、weight_norm 移除 |
| GeneratorLoRA Creation | ✅ | 无 LoRA 创建、前向传播 |
| GeneratorLoRA with LoRA | ✅ | LoRA 注入、参数统计、前向传播 |
| SynthesizerLoRA Wrapper | ✅ | 包装器功能、forward/infer |
| Generator ResBlock Types | ✅ | ResBlock1 和 ResBlock2 支持 |
| LoRA Parameter Freezing | ✅ | 参数冻结验证 |

## 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `models/__init__.py` | 22 | 模块导出 |
| `models/resblock.py` | 233 | ResBlock1, ResBlock2 实现 |
| `models/generator_lora.py` | 282 | GeneratorLoRA 实现 |
| `models/synthesizer_lora.py` | 310 | SynthesizerLoRA 实现 |
| `tests/test_model.py` | 410 | 模型集成测试 |

**总代码行数**: ~1,257 行

## API 参考

### GeneratorLoRA

```python
class GeneratorLoRA(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,  # "1" or "2"
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
        lora_config: Optional[LoRAConfig] = None,
    )

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ) -> torch.Tensor

    def remove_weight_norm(self)
```

### SynthesizerLoRA

```python
class SynthesizerLoRA(nn.Module):
    def __init__(
        self,
        base_synthesizer: nn.Module,
        lora_config: Optional[LoRAConfig] = None,
        freeze_non_lora: bool = True,
    )

    def forward(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds=None)
    def infer(self, phone, phone_lengths, pitch, nsff0, sid, ...)
    def get_lora_parameters(self) -> List[nn.Parameter]
    def get_lora_state_dict(self) -> dict
    def load_lora_weights(self, lora_state_dict: dict, strict: bool = True)
    def unfreeze_all(self)
    def freeze_base_model(self)
```

## 下一步

阶段 3 将实现训练流程：
- 训练脚本 (`training/train_lora.py`)
- 数据加载器 (`training/data_loader.py`)
- 损失函数 (`training/losses.py`)
- 训练配置和日志

---

**完成日期**: 2026-01-28
**状态**: ✅ 已完成
