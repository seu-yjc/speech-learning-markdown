---
title: GTCRN
date: 2024-03-01
tags:
  - speech-enhancement
  - lightweight-model
  - causal
  - streaming
  - ICASSP2024
  - ultralow-computation
aliases:
  - Grouped Temporal Convolutional Recurrent Network
  - 分组时序卷积循环网络
---

# GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources

## 论文信息

- **作者**：Xiaobin Rong, Tianchi Sun, Xu Zhang, Yuxiang Hu, Changbao Zhu, Jing Lu†
- **单位**：Key Laboratory of Modern Acoustics, Nanjing University & NJU-Horizon Intelligent Audio Lab, Horizon Robotics & Jiangsu Thingstar Information Technology Co., Ltd.
- **发表**：**ICASSP 2024**
- **代码/Demo**：[https://github.com/Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)
- **部署生态**：PyTorch / ONNX / LADSPA / sherpa-onnx / Web (WASM) / TensorRT

## 摘要

现代深度学习语音增强模型虽然性能优异，但往往需要大量参数和计算资源，难以部署到边缘设备上。本文提出 **GTCRN（Grouped Temporal Convolutional Recurrent Network）**，以 DPCRN 为骨干，通过**分组策略**大幅压缩模型规模，并结合**子带特征提取（SFE）**模块和**时序递归注意力（TRA）**模块提升性能。最终模型仅需 **23.7 K 参数**和 **39.6 MMACs/s** 的计算量，不仅大幅超越同量级的 RNNoise，还能与计算量高几个数量级的基线模型竞争。

> [!note] 核心创新
> GTCRN 将 ShuffleNetV2 的高效 CNN 设计范式引入语音增强领域，结合 ERB 感知频带压缩、分组卷积/RNN、子带特征提取和时序递归注意力，在**极致轻量**的约束下实现高性能语音增强，且保证**完全因果性**，支持逐帧流式推理。

## 核心贡献

1. **极致轻量化**：仅 **23.7 K** 可学习参数（含 ERB 为 48.2 K）、**39.6 MMACs/s**（修正后 33.0 MMACs/s），比 RNNoise（60 K）还小，性能却大幅超越

2. **分组策略降低复杂度**：将分组卷积（ShuffleNetV2）和分组RNN引入DPCRN骨干，高效压缩模型

3. **SFE + TRA 模块**：子带特征提取将频带关系整合到通道维度，时序递归注意力通过 GRU 建模能量包络——两个轻量模块增量式提性能

4. **完全因果 + 流式推理**：Inter-RNN 单向、时域因果卷积、状态缓存机制，支持逐帧处理，RTF = 0.07（Intel i5-12400 CPU）

5. **完善开源生态**：提供 PyTorch / ONNX / LADSPA / sherpa-onnx / Web（WASM）/ TensorRT 多种部署方案

> [!note] BM: band merging, BS: band splitting, G-DPRNN: grouped dual-path RNN
> PDF: {{pdfAttachments}}
> 库: {{localLibrary}}
> 白板:

> [!info]
> {{DOI}}
> {{publicationTitle}} — ICASSP 2024: "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources"
> {{author}} — Xiaobin Rong et al.
> {{date}}
> {{tags}} #speech-enhancement #lightweight #causal #streaming #ICASSP2024

## 整体 Pipeline

```
波形 → STFT(512, 256, sqrt-hann) → (B, F, T, 2) 复数谱
  → 拆为 [mag, real, imag] 三通道 → (B, 3, T, 257)
  → ERB-BM (Band Merging): 257 bins → 129 bins (65 low + 64 ERB-compressed high)
  → SFE (Subband Feature Extraction): unfold k=3 → (B, 9, T, 129)
  → Encoder (5层, 逐步降频: 129→65→33)
  → 2× G-DPRNN (时序建模)
  → Decoder (5层, 逐步升频: 33→65→129, skip connections)
  → ERB-BS (Band Splitting): 129 bins → 257 bins
  → CRM (Complex Ratio Mask) 与原始复数谱相乘
  → iSTFT → 增强波形
```

**关键参数**：
- STFT: nfft=512, hop_length=256, win_length=512, window=sqrt(hann)
- 输入维度: `(B, F=257, T, 2)` → 三通道拼接 `(B, 3, T, 257)`
- 采样率: 16 kHz

## 网络架构

GTCRN 由 **BM/BS 模块**、**SFE 模块**（可选）、**编码器**、**G-DPRNN** 和**解码器**组成。编码器包含 2 个 Conv 块 + 3 个 GT-Conv 块，解码器为镜像结构，通过 skip connection 连接。

### 1. ERB 模块 — 频带压缩 / 恢复

基于等效矩形带宽（Equivalent Rectangular Bandwidth）的非均匀频率压缩。

**动机**：人耳对低频的分辨率高于高频；低频保留更多 bin，高频用 ERB 尺度压缩以减少计算量。

**实现** (`ERB(erb_subband_1=65, erb_subband_2=64)`)：
- nfft=512 → 257 个频带
- **低频区（0~2kHz）**：前 65 个频带直接保留（不做压缩）
- **高频区（2k~8kHz）**：剩余 192 个频带通过 ERB 三角滤波器组压缩到 64 个
- 总计：65 + 64 = 129 个频带

**BM (Band Merging)**：
```python
x_low = x[..., :65]           # 直接保留低频
x_high = erb_fc(x[..., 65:])  # 192 → 64, 固定矩阵乘法
return cat([x_low, x_high])   # → 129 bins
```

**BS (Band Splitting)**：BM 的逆操作，`ierb_fc` 使用滤波器矩阵的转置 (`erb_filters.T`)。

**三角滤波器构造**：
- 将目标 ERB 频率范围等分，映射回 Hz 域
- 相邻滤波器之间有 50% 重叠（三角窗）
- 滤波器权重固定（**不可学习**），存储在 `nn.Linear` 的 weight 中（`requires_grad=False`）

> [!warning] 复杂度修正
> 论文报告 23.7K 参数 / 39.6 MMACs，但若计入 ERB 的不可学习参数，实际为 48.2K 参数。此外，将低频区的矩阵乘法替换为简单拼接后，MACs 降至 **33.0 MMACs**。

### 2. SFE — Subband Feature Extraction（子带特征提取）

```python
class SFE:
    def __init__(self, kernel_size=3, stride=1):
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size),
                                stride=(1, stride),
                                padding=(0, (kernel_size-1)//2))

    def forward(self, x):  # x: (B, C, T, F)
        xs = self.unfold(x)  # 沿频率维展开
        # reshape: (B, C*k, T, F)
        return xs
```

![[Pasted image 20260605173103.png]]

**核心思想**：使用 `nn.Unfold` 在频率维度上以滑动窗口方式展开，将每个频带与其上下相邻的 k-1 个频带组合，沿通道维度堆叠。**将频率维度上的子带关系整合到通道维度中**，使后续的逐点卷积能够捕捉跨频带交互。

- k=3 时，每个频带与其 ±1 邻居组合 → 通道数从 C 变为 3C
- 这是 GTConvBlock 的输入预处理，替代了标准卷积中的频率维卷积

### 3. TRA — Temporal Recurrent Attention（时序递归注意力）

![[Pasted image 20260605173122.png]]

```python
class TRA:
    def __init__(self, channels):
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, T, F)
        zt = mean(x², dim=-1)          # (B, C, T) 每通道能量
        at = GRU(zt.transpose(1,2))     # (B, T, C*2) 时序建模
        at = fc(at).transpose(1,2)      # (B, C, T)
        at = sigmoid(at)[..., None]     # (B, C, T, 1) 注意力权重
        return x * at                    # 通道级门控
```

**数学定义**：给定输入特征 $\mathbf{V} \in \mathbb{R}^{C \times T \times F}$，先通过全局平均池化计算时序能量表示 $\mathbf{Z} \in \mathbb{R}^{C \times T}$：

$$\mathbf{Z}(c, t) = \frac{1}{F} \sum_{f=1}^{F} \mathbf{V}^2(c, t, f)$$

然后通过 GRU + FC + Sigmoid 生成注意力掩码 $\mathbf{A} \in \mathbb{R}^{C \times T \times F}$，最终输出 $\tilde{\mathbf{V}} = \mathbf{V} \otimes \mathbf{A}$。

**作用**：在时间维度上对每个通道计算能量包络，通过 GRU 建模时序依赖，生成通道级注意力门控。这是一种轻量级的时序注意力机制，仅增加极少参数。

### 4. GTConvBlock — Grouped Temporal Convolution Block（核心模块）

![[Pasted image 20260605173145.png]]

基于 **ShuffleNetV2** 单元设计，结合 SFE + TRA。

```
输入 x: (B, C, T, F)
    ↓
Channel Split: x1, x2 = chunk(x, 2, dim=1)  → 各 C/2 通道
    ↓
x1 分支 (处理分支):
    SFE(k=3)          → (B, C/2*3, T, F)
    PointConv1 (1×1)  → (B, hidden, T, F) + BN + PReLU
    DepthConv (kT×kF) → (B, hidden, T, F) + BN + PReLU  [分组卷积=深度可分离]
    PointConv2 (1×1)  → (B, C/2, T, F) + BN
    TRA               → (B, C/2, T, F)  时序注意力
    ↓
Channel Shuffle: shuffle(x1_processed, x2)  → (B, C, T, F)
```

**关键设计**：
- **Channel Split + Shuffle**：ShuffleNetV2 的核心思想。一半通道直接旁路（类似残差），另一半经过深度可分离卷积处理，最后通过 channel shuffle 融合信息
- **深度可分离卷积**：`groups=hidden_channels`，大幅减少参数和计算量
- **因果膨胀卷积**：时间维使用 dilation（1/2/5），padding 仅在时间维左侧（`self.pad_size, 0`），确保因果性
- **TRA 加在 x1 分支末**：在融合前对处理后的特征施加时序注意力

**多尺度时序感受野**：Encoder/Decoder 中使用 dilation=1, 2, 5 的 GTConvBlock 堆叠，逐步增大时间感受野。

### 5. GRNN — Grouped RNN

将标准 RNN 分解为两个独立的小 RNN，分别处理分组后的特征。

```python
class GRNN:
    def __init__(self, input_size, hidden_size, ...):
        # 两个独立 GRU，各处理一半
        self.rnn1 = GRU(input_size//2, hidden_size//2, ...)
        self.rnn2 = GRU(input_size//2, hidden_size//2, ...)

    def forward(self, x, h):
        x1, x2 = chunk(x, 2, dim=-1)   # 特征分组
        h1, h2 = chunk(h, 2, dim=-1)   # 隐藏状态分组
        y1, h1 = rnn1(x1, h1)
        y2, h2 = rnn2(x2, h2)
        return cat([y1, y2]), cat([h1, h2])
```

> [!tip] 参数效率
> 将一个大 RNN（参数量 ∝ hidden²）拆为两个小 RNN（参数量 ∝ 2×(hidden/2)² = hidden²/2），减少约 **一半参数**。同时分组独立计算可并行执行。

> [!note] 关于显式特征重排
> 论文中原本有显式的特征重排层（feature shuffle），但会使模型不可流式处理。实际代码中**丢弃了显式重排**，通过 G-DPRNN 中后续的 FC 层隐式实现特征交互。

### 6. G-DPRNN — Grouped Dual-Path RNN

```
输入 x: (B, C, T, F=33)
    ↓
=== Intra-RNN (帧内建模, 沿频率轴) ===
    permute → (B, T, F, C)
    reshape → (B*T, F, C)
    GRNN (bidirectional=True)  → (B*T, F, C)
    FC + LayerNorm + Residual
    ↓
=== Inter-RNN (帧间建模, 沿时间轴) ===
    permute → (B, F, T, C)
    reshape → (B*F, T, C)
    GRNN (bidirectional=False) → (B*F, T, C)   ← 单向，保证因果性！
    FC + LayerNorm + Residual
    ↓
输出: (B, C, T, F)
```

**核心设计**：
- **Intra-RNN**：双向 GRNN，沿频率轴建模谱间依赖关系（帧内）
- **Inter-RNN**：**单向** GRNN，沿时间轴建模时序依赖（帧间）→ **保证因果性**
- DPRNN 通过分两路交替建模，将 $\mathcal{O}(T \times F)$ 的全连接序列建模降为 $\mathcal{O}(T + F)$

### 7. Encoder / Decoder

**Encoder**（5层，逐层降频）：

| 层 | 类型 | 输入→输出通道 | Kernel | Stride | Groups | Dilation | 频率维度变化 |
|----|------|-------------|--------|--------|--------|----------|------------|
| 1 | ConvBlock | 9→16 | (1, 5) | (1, 2) | 1 | 1 | 129→65 |
| 2 | ConvBlock | 16→16 | (1, 5) | (1, 2) | 2 | 1 | 65→33 |
| 3 | GTConvBlock | 16→16 | (3, 3) | (1, 1) | - | 1 | 33 |
| 4 | GTConvBlock | 16→16 | (3, 3) | (1, 1) | - | 2 | 33 |
| 5 | GTConvBlock | 16→16 | (3, 3) | (1, 1) | - | 5 | 33 |

- 前两层用普通 ConvBlock 做频率下采样（stride=2），后三层为 GTConvBlock（不改变尺寸，通过 dilation 扩增时序感受野）
- ConvBlock: Conv → BatchNorm → PReLU（最后一层为 Tanh）
- 第2层使用 groups=2 的分组卷积进一步减少参数

**Decoder**（5层，逐层升频，镜像结构）：

| 层 | 类型 | 输入→输出通道 | Kernel | Stride | Dilation |
|----|------|-------------|--------|--------|----------|
| 1 | GTConvBlock (deconv) | 16→16 | (3, 3) | (1, 1) | 5 |
| 2 | GTConvBlock (deconv) | 16→16 | (3, 3) | (1, 1) | 2 |
| 3 | GTConvBlock (deconv) | 16→16 | (3, 3) | (1, 1) | 1 |
| 4 | ConvBlock (deconv) | 16→16 | (1, 5) | (1, 2) | 1 |
| 5 | ConvBlock (deconv) | 16→2 | (1, 5) | (1, 2) | 1 |

- 使用转置卷积（ConvTranspose2d）上采样恢复频率分辨率
- **Skip Connections**：每层 Decoder 输入 = 上层输出 + 对应 Encoder 层的输出（`x + en_outs[N-1-i]`）
- 最后一层输出 2 通道 → CRM 的实部和虚部，激活函数为 **Tanh**（限制 mask 值在 [-1, 1]）

### 8. CRM — Complex Ratio Mask（复数比掩膜）

```python
class Mask:
    def forward(self, mask, spec):  # mask: (B,2,T,F), spec: (B,2,T,F)
        s_real = spec[:,0]*mask[:,0] - spec[:,1]*mask[:,1]
        s_imag = spec[:,1]*mask[:,0] + spec[:,0]*mask[:,1]
        # 等价于复数乘法: S_enhanced = S_noisy ⊙ M
```

CRM 同时估计幅度和相位的修正，相比仅估计幅度掩膜（IRM）能更好地恢复语音相位信息。与 DCCRN-C 的模式一致（遵循复数乘法规则）。

## 损失函数 — Hybrid Loss

损失函数联合优化**波形域**和**频谱域**：

$$\mathcal{L} = \alpha \mathcal{L}_{\text{SISNR}}(\tilde{\mathbf{s}}, \mathbf{s}) + (1 - \beta) \mathcal{L}_{\text{mag}}(\tilde{\mathbf{S}}, \mathbf{S}) + \beta \left[ \mathcal{L}_{\text{real}}(\tilde{\mathbf{S}}, \mathbf{S}) + \mathcal{L}_{\text{imag}}(\tilde{\mathbf{S}}, \mathbf{S}) \right]$$

其中 $\alpha = 0.01$，$\beta = 0.3$。各项定义如下：

$$\mathcal{L}_{\text{SISNR}} = -10 \log_{10} \left( \frac{\|\mathbf{s}_t\|^2}{\|\tilde{\mathbf{s}} - \mathbf{s}_t\|^2} \right); \quad \mathbf{s}_t = \frac{\langle \tilde{\mathbf{s}}, \mathbf{s} \rangle \mathbf{s}}{\|\mathbf{s}\|^2}$$

$$\mathcal{L}_{\text{mag}}(\tilde{\mathbf{S}}, \mathbf{S}) = \text{MSE}(|\tilde{\mathbf{S}}|^{0.3}, |\mathbf{S}|^{0.3})$$

$$\mathcal{L}_{\text{real}}(\tilde{\mathbf{S}}, \mathbf{S}) = \text{MSE}(\tilde{\mathbf{S}}_r / |\tilde{\mathbf{S}}|^{0.7}, \mathbf{S}_r / |\mathbf{S}|^{0.7})$$

$$\mathcal{L}_{\text{imag}}(\tilde{\mathbf{S}}, \mathbf{S}) = \text{MSE}(\tilde{\mathbf{S}}_i / |\tilde{\mathbf{S}}|^{0.7}, \mathbf{S}_i / |\mathbf{S}|^{0.7})$$

> [!important] Power-law 压缩
> 对幅度取 **0.3 次幂**、实/虚部除以幅度的 0.7 次幂后进行 MSE 计算，平衡大能量和小能量频带的贡献。联合**频域（mag + complex）+ 时域（SISNR）**优化，兼顾幅度和相位。

等价于代码中的加权形式：`HybridLoss = 30×(real_loss + imag_loss) + 70×mag_loss + SISNR`

## 因果性与流式推理

### 因果性保证

1. **Inter-RNN 使用单向 GRU**：只依赖过去帧
2. **时域卷积使用 causal padding**：只在时间维左侧补零，不使用未来帧
3. **Intra-RNN 使用双向 GRU**：沿频率轴，不影响因果性（同一帧内）
4. 代码中有 causality check：输入前半部分固定，后半部分不同，验证前半部分输出完全一致

### 流式推理（Streaming Inference）

`stream/` 目录下的实现支持逐帧流式处理：

- **StreamConv2d / StreamConvTranspose2d**：维护时间维的卷积缓存（cache），每步只处理新到达的 1 帧
- **StreamTRA**：维护 GRU 的隐藏状态缓存
- **G-DPRNN**：维护 Inter-RNN 的隐藏状态缓存（Intra-RNN 每帧独立计算）
- **缓存形状**：
  - `conv_cache`: (2, 1, 16, 16, 33) — Encoder + Decoder 的卷积缓存
  - `tra_cache`: (2, 3, 1, 1, 16) — 3个 GTConvBlock 的 TRA-GRU 状态
  - `inter_cache`: (2, 1, 33, 16) — 2个 DPGRNN 的 Inter-RNN 状态
- 导出为 ONNX 格式，通过 ONNX Runtime 推理
- **RTF = 0.07** on Intel i5-12400 CPU @ 2.50 GHz

### 部署支持

- **LADSPA 插件**：Linux PipeWire 实时音频滤波（社区贡献）
- **sherpa-onnx**：集成到 sherpa-onnx 推理框架
- **Web demo**：基于 WASM 的浏览器端推理（HuggingFace Space）
- **TRT-SE**：TensorRT 部署示例

## 实验设置

### 数据集

**VCTK-DEMAND 数据集**：
- 训练集：11,572 条语音（28 个说话人），其中 1,572 条用于验证
- 测试集：824 条语音（2 个说话人）
- 预混合的带噪-干净语音对
- 采样率重采样至 16 kHz

**DNS3 大规模数据集**（主要训练集）：
- 包含大量干净语音集、噪声集和 RIR
- 额外加入 DiDiSpeech 中文语料库
- 动态混合：干净语音卷积随机 RIR + 随机噪声片段，SNR 范围 **-5 至 15 dB**
- 训练目标：保留前 100 ms 反射声
- 总计生成 **720,000 对** 10 秒带噪-干净数据
- 验证集：840 对，测试集：800 对
- 同时在 DNS Challenge 3 盲测集上评估

### 训练设置

| 参数 | 配置 |
|------|------|
| 窗长 / 帧移 | 32 ms / 16 ms |
| FFT 长度 | 512 |
| 窗函数 | sqrt(Hann) |
| 采样率 | 16 kHz |
| 输入特征 | [mag, real, imag] 三通道拼接 |
| ERB 配置 | 低频谱 65 bins + 高频谱 192→64 bins = 129 bins |
| 优化器 | Adam |
| 初始学习率 | 0.001（连续 5 epoch 验证 loss 不降则减半） |
| Batch Size | 4（VCTK-DEMAND）/ 16（DNS3） |
| 训练段长度 | 8 秒（DNS3），每 epoch 随机选 40,000 对 |
| 框架 | PyTorch |

### 基线模型

| 模型 | 参数量 | MACs | 特点 |
|------|:---:|:---:|------|
| RNNoise (2018) | 60 K | 0.04 G/s | 经典轻量模型，DSP+DL 混合 |
| PercepNet (2020) | 8.0 M | 0.80 G/s | 感知驱动的低复杂度增强 |
| DeepFilterNet (2022) | 1.8 M | 0.35 G/s | 基于深度滤波的全带增强 |
| S-DCCRN (2022) | 2.34 M | - | 超宽带 DCCRN 变体 |
| **GTCRN** | **23.7 K** | **0.04 G/s** | 本文模型 |

## 实验结果

![[Pasted image 20260605173310.png]]
![[Pasted image 20260605173321.png]]
![[Pasted image 20260605173327.png]]

### 消融实验（DNS3 测试集）

> [!important] SFE + TRA 消融

| SFE | TA | TRA | 参数量 (K) | MACs (M/s) | SISNR | PESQ | STOI |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| - | - | - | - | - | 3.92 | 1.30 | 0.789 |
| ✗ | ✗ | ✗ | 13.35 | 33.91 | 9.87 | 1.87 | 0.834 |
| ✗ | ✓ | ✗ | 14.84 | 34.00 | 10.00 | 1.89 | 0.838 |
| ✗ | ✗ | ✓ | 21.65 | 34.47 | 10.25 | 1.91 | 0.840 |
| ✓ | ✗ | ✗ | 15.37 | 39.07 | 10.10 | 1.90 | 0.838 |
| ✓ | ✓ | ✗ | 16.86 | 39.16 | 10.29 | 1.92 | 0.841 |
| ✓ | ✗ | **✓** | **23.67** | **39.63** | **10.39** | **1.94** | **0.844** |

**分析**：
- **TRA 优于 TA**：时序递归注意力比普通时间注意力效果更好，且计算增量极小
- **SFE + TRA 联合最优**：两个模块产生协同效应，所有指标均达最佳
- 从基线（无 SFE/TRA）到全模块，PESQ 提升 0.07，SISNR 提升 0.52 dB

### VCTK-DEMAND 测试集结果

> [!important] 与基线模型对比

| 模型 | 参数量 (M) | MACs (G/s) | SISNR | PESQ | STOI |
|------|:---:|:---:|:---:|:---:|:---:|
| Noisy | - | - | 8.45 | 1.97 | 0.921 |
| RNNoise (2018) | 0.06 | 0.04 | - | 2.29 | - |
| PercepNet (2020) | 8.00 | 0.80 | - | 2.73 | - |
| DeepFilterNet (2022) | 1.80 | 0.35 | 16.63 | 2.81 | 0.942 |
| S-DCCRN (2022) | 2.34 | - | - | 2.84 | 0.940 |
| **GTCRN** | **0.02** | **0.04** | **18.83** | **2.87** | **0.940** |

**分析**：
- GTCRN 以 **仅 20 K 参数**在 SISNR 和 PESQ 上**全面超越**所有基线模型
- 相比同量级的 RNNoise（60 K），PESQ 从 2.29 提升至 2.87（+0.58），提升巨大
- 相比大 90 倍的 DeepFilterNet（1.8 M），PESQ 仍高出 0.06

### DNS3 盲测集结果

> [!important] DNSMOS 主观质量评分

| 模型 | 参数量 (M) | MACs (G/s) | DNSMOS-P.808 | DNSMOS-P.835 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | **BAK** | **SIG** | **OVRL** |
| Noisy | - | - | 2.96 | 2.65 | 3.20 | 2.33 |
| RNNoise (2018) | 0.06 | 0.04 | 3.15 | 3.45 | 3.00 | 2.53 |
| S-DCCRN (2022) | 2.34 | - | 3.43 | - | - | - |
| **GTCRN** | **0.02** | **0.04** | **3.44** | **3.90** | **3.00** | **2.70** |

**分析**：
- GTCRN 在所有 DNSMOS 指标上**全面超越 RNNoise**，尤其在信号质量（SIG=3.90）上提升显著
- 甚至超越了参数量大 100 倍的 S-DCCRN（2.34 M）
- 以 20 K 参数达到媲美大模型的语音质量

### 推理速度

在 Intel i5-12400 CPU @ 2.50 GHz 上，GTCRN 的流式推理 **RTF = 0.07**，完全满足实时处理需求。

## 消融实验关键结论

- **SFE 模块**：验证了将子带关系整合到通道维度的有效性——使 Conv1×1 也能利用跨频带信息
- **TRA 模块**：验证了时序递归注意力的增益——GRU 建模能量包络优于简单的时间平均注意力
- **ShuffleNetV2 结构**：Channel split + shuffle 在极低参数量下信息融合的有效性
- **ERB 压缩**：利用听觉感知先验，"免费"减少约 50% 的频率维度计算量

## 后续工作 & 生态

- **UL-UNAS**（2025.03）：改进的超轻量 SE 模型，使用 NAS 搜索架构
- **H-GTCRN**（2025.05）：适用于低 SNR 场景的轻量混合双通道 SE 系统

## 结论与展望

**结论**：
1. GTCRN 通过对 DPCRN 应用分组卷积、分组 RNN、ERB 频带压缩等多重策略，将模型压缩至 **23.7 K 参数 / 39.6 MMACs/s**
2. SFE + TRA 两个轻量模块在几乎不增加计算量的前提下，持续提升性能
3. 在 VCTK-DEMAND 和 DNS3 数据集上，不仅大幅超越同量级 RNNoise，还能与计算量高几个数量级的模型竞争

**展望**（论文未明确，基于方向推断）：
- 将 GTCRN 部署到助听器、TWS 耳机等**超低功耗边缘设备**
- 结合 NAS 进一步自动化架构搜索（→ UL-UNAS）
- 多通道扩展以应对更复杂的声学场景（→ H-GTCRN）

## 相关文献

- [[DPCRN]] — Le et al., "DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement," Interspeech 2021
- [[DCCRN]] — Hu et al., "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement," Interspeech 2020
- [[ShuffleNetV2]] — Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," ECCV 2018
- [[RNNoise]] — Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement," MMSP 2018
- [[DeepFilterNet]] — Schröter et al., "DeepFilterNet: A Low Complexity Speech Enhancement Framework," ICASSP 2022
- [[S-DCCRN]] — Lv et al., "S-DCCRN: Super Wide Band DCCRN with Learnable Complex Feature," ICASSP 2022

## 亮点

1. **极致轻量**：仅 48.2K 参数 / 33 MMACs，比 RNNoise（60K）还小，性能却大幅超越
2. **完全因果**：适用于实时语音增强场景（助听器、耳机、VoIP）
3. **ShuffleNetV2 + SFE + TRA**：将高效 CNN 设计引入语音增强，每个模块增量式地提升性能
4. **ERB 非均匀频带压缩**：利用听觉感知特性减少计算量，低频高分辨率 + 高频低分辨率
5. **完善的开源生态**：提供 PyTorch / ONNX / LADSPA / sherpa-onnx / Web 等多种部署方案
6. **Hybrid Loss**：频域（mag + complex）+ 时域（SISNR）联合优化，兼顾幅度和相位

## 思考与启发

1. **ShuffleNetV2 单元可用于语音增强**：在极低参数量约束下，channel split + depthwise conv + shuffle 是高效的架构选择
2. **ERB 尺度压缩是"免费"的**：利用听觉感知先验，用不可学习的固定滤波器压缩频带，几乎不增加计算量
3. **分组 RNN 是参数高效选择**：将大 RNN 拆为多个小 RNN，近似效果但参数仅为一半
4. **Intra+Inter DPRNN 分离建模**：帧内用双向（频率相关性）、帧间用单向（因果时序），设计精巧
5. **CRM 优于 IRM**：同时估计幅度和相位修正，在极轻量模型中也能获得更好的语音质量
6. **流式推理关键在于状态缓存**：卷积缓存 + RNN 隐状态缓存，避免了逐帧推理时的重复计算
7. **与 DCCRN 的对比启示**：DCCRN（3.7M 参数）追求复数建模精度，GTCRN（23.7K 参数）追求计算效率极致——两条不同的优化路径，分别适用于不同场景。GTCRN 证明**极致轻量 ≠ 牺牲性能**
