# RVC-LoRA 快速入门指南

> 5 分钟快速上手 RVC-LoRA 语音转换

---

## 📋 前置要求

- Python 3.8+
- 5-10 分钟的清晰语音数据
- （可选）NVIDIA GPU 用于加速训练

---

## 🚀 快速开始

### 步骤 1: 安装依赖

```bash
# 进入 LoraModel 目录
cd LoraModel

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 下载预训练模型

```bash
# 下载所有必需的模型（约 300 MB）
python download_models.py --all

# 验证下载
python download_models.py --check
```

下载的模型包括：
- ✅ HuBERT 模型 (189 MB) - 用于特征提取
- ✅ RMVPE 模型 (55 MB) - 用于音高提取
- ✅ Pretrained V2 模型 (每个约 55 MB) - 基础模型

### 步骤 3: 准备训练数据

将你的语音文件放在一个文件夹中：

```
my_voice/
├── audio1.wav
├── audio2.wav
├── audio3.wav
└── ...
```

**数据要求**：
- 格式：WAV、MP3、FLAC 等（会自动转换）
- 时长：建议至少 5-10 分钟
- 质量：清晰、低噪音的录音

### 步骤 4: 训练 LoRA 模型

```bash
# 基本训练命令
python scripts/train_lora_e2e.py \
    --input_dir ./my_voice \
    --output_dir ./output \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --epochs 100
```

**训练参数说明**：
- `--input_dir`: 包含音频文件的文件夹
- `--output_dir`: 输出文件夹（保存 LoRA 权重）
- `--base_model`: 基础模型路径
- `--epochs`: 训练轮数（100 是推荐值）

**可选参数**：
```bash
python scripts/train_lora_e2e.py \
    --input_dir ./my_voice \
    --output_dir ./output \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --epochs 100 \
    --batch_size 4 \          # 批次大小（显存不足时减小）
    --lora_rank 8 \           # LoRA 秩（4-16）
    --sample_rate 40000       # 采样率（32000/40000/48000）
```

**训练时间**：
- CPU: 2-4 小时（100 epochs）
- GPU (RTX 3090): 30-60 分钟（100 epochs）

**训练输出**：
```
output/
├── lora_final.pth          # 最终的 LoRA 权重
├── lora_best.pth           # 验证集上最好的权重
├── checkpoints/            # 训练检查点
├── logs/                   # TensorBoard 日志
└── train_*.log             # 训练日志
```

### 步骤 5: 使用 LoRA 进行推理

```bash
# 基本推理命令
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth
```

**音高调整**（可选）：
```bash
# 升高 2 个半音（男声转女声）
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth \
    --pitch 2

# 降低 2 个半音（女声转男声）
python scripts/infer_lora_e2e.py \
    --source ./input.wav \
    --output ./output.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./output/lora_final.pth \
    --pitch -2
```

---

## 🎯 完整示例

从零开始训练和使用 LoRA 模型：

```bash
# 1. 下载模型
python download_models.py --all

# 2. 训练 LoRA（使用示例数据）
python scripts/train_lora_e2e.py \
    --input_dir ./download/base_voice \
    --output_dir ./my_first_lora \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --epochs 50

# 3. 测试推理
python scripts/infer_lora_e2e.py \
    --source ./download/test_voice/7.wav \
    --output ./converted.wav \
    --model ./download/pretrained_v2/f0G40k.pth \
    --lora ./my_first_lora/lora_final.pth

# 4. 播放结果
# macOS: afplay ./converted.wav
# Linux: aplay ./converted.wav
# Windows: start ./converted.wav
```

---

## 📊 监控训练进度

使用 TensorBoard 查看训练曲线：

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/logs

# 在浏览器中打开 http://localhost:6006
```

---

## 🔧 常见问题

### Q: 显存不足怎么办？

减小 batch_size：
```bash
python scripts/train_lora_e2e.py \
    --input_dir ./my_voice \
    --output_dir ./output \
    --base_model ./download/pretrained_v2/f0G40k.pth \
    --batch_size 2  # 或者 1
```

### Q: 训练速度太慢？

1. 确保使用 GPU（检查 `torch.cuda.is_available()`）
2. 使用较低的采样率：`--sample_rate 32000`
3. 减少训练轮数：`--epochs 50`

### Q: 转换质量不好？

1. 增加训练数据（至少 10 分钟）
2. 增加训练轮数（100-200 epochs）
3. 调整音高：`--pitch 2` 或 `--pitch -2`
4. 使用更高质量的训练数据

### Q: 如何选择采样率？

- **32000 Hz**: 速度快，质量一般
- **40000 Hz**: 推荐，平衡质量和速度
- **48000 Hz**: 质量最高，速度较慢

---

## 📚 下一步

- 阅读完整文档：[README.md](README.md)
- 查看项目进度：[PROGRESS.md](PROGRESS.md)
- 了解技术细节：[PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)
- 查看待办事项：[TODO.md](TODO.md)

---

## 💡 提示

1. **数据质量 > 数据数量**：10 分钟高质量录音比 1 小时低质量录音效果更好
2. **从小开始**：先用 50 epochs 快速测试，满意后再用 100-200 epochs 完整训练
3. **保存检查点**：训练会自动保存检查点，可以随时中断和恢复
4. **实验音高**：不同的音高调整值会产生不同的效果，多试几个值

---

## 🆘 获取帮助

遇到问题？
1. 查看 [README.md](README.md) 中的"故障排除"部分
2. 查看 [TODO.md](TODO.md) 中的已知问题
3. 提交 Issue 到 GitHub

---

**祝你使用愉快！🎉**
