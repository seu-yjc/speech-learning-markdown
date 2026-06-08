---
title: DCCRN
date: 2020-08-02
tags:
  - speech-enhancement
  - deep-learning
  - complex-network
  - phase-aware
  - Interspeech2020
aliases:
  - Deep Complex Convolution Recurrent Network
  - 复数卷积循环网络
---

# DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement

## 论文信息

- **作者**：Yanxin Hu, Yun Liu*, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang, Lei Xie†（* 共同一作，† 通讯作者）
- **单位**：Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University, Xi'an & AI Interaction Division, Sogou Inc., Beijing
- **发表**：arXiv:2008.00264v4，收录于 **Interspeech 2020**
- **代码/Demo**：[https://huyanxin.github.io/DeepComplexCRN](https://huyanxin.github.io/DeepComplexCRN)

## 摘要

语音增强得益于深度学习在可懂度和感知质量方面的成功。传统时频域方法侧重于通过CNN或RNN预测TF掩码或语音频谱。一些近期研究使用**复数谱**作为训练目标，但仍在**实值网络**中训练，分别预测幅度/相位或实部/虚部。CRN集成了CED结构和LSTM，已被证明对复数目标有帮助。为了更有效地训练复数目标，本文设计了**DCCRN（Deep Complex Convolution Recurrent Network）**，使CNN和RNN结构均能处理复数运算。

> [!note] 核心创新
> DCCRN 在编码器/解码器中引入**复数Conv2d、复数Batch Normalization和复数LSTM**，通过模拟复数乘法规则建模幅度和相位之间的**相关性**，而非将实部和虚部作为两个独立通道处理。

## 核心贡献

1. **方法设计**：DCCRN 结合了卷积编码器-解码器架构与 <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2Flocal%2F09UGClb5%2Fitems%2FQNRA3Y3H%22%2C%22pageLabel%22%3A%221%22%2C%22position%22%3A%7B%22pageIndex%22%3A0%2C%22rects%22%3A%5B%5B57.60000000000022%2C461.0622576000003%2C145.35415680000025%2C468.8361264000003%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2Flocal%2F09UGClb5%2Fitems%2FXXHUSN6D%22%5D%2C%22locator%22%3A%221%22%7D%7D" ztype="zhighlight"><a href="zotero://open/library/items/QNRA3Y3H?page=1">LSTM</a></span>，引入了**复数卷积、复数批量归一化和复数LSTM**模块，模拟复数乘法规则处理时频域信息，从而同时捕捉语音的幅度与相位关联。与仅用实值网络的传统方法相比，这种设计增强了对复数目标的建模能力。

2. **训练目标**：网络以**复数比值掩码（CRM）**作为核心训练目标，通过信号逼近（SA）损失函数直接最小化干净语音与增强语音的复数谱差异。同时探索了三种掩码模式——**DCCRN-R**（实虚部分别估计）、**DCCRN-C**（笛卡尔坐标下的CSA方式）和**DCCRN-E**（极坐标，tanh限制幅度到0-1），证实复数目标更具性能优势。

3. **实验表现**：在模拟的 WSJ0 数据集上，DCCRN-CL（含复数 LSTM 的版本）在 PESQ 得分上显著优于传统 LSTM 和 CRN 模型，且仅用 **3.7M 参数**，计算复杂度约为 DCUNET 的 **1/6**。在 **Interspeech 2020 DNS 挑战赛**中，该模型以 3.7M 参数在实时轨道上获得 **MOS 第一名**（4.00@no reverb），非实时轨道**排名第二**。

4. **应用价值**：DCCRN 通过降低模型大小和计算成本，为实时语音增强场景（如边缘设备）提供了高效方案，并在回声和混响条件下展现出更稳定的增强能力。

主要研究方向：基于深度学习的**单通道语音增强**，追求**低模型复杂度和实时处理**。

## 网络架构

### 整体架构

DCCRN 本质上是一种**因果CED架构**，在编码器和解码器之间具有两个LSTM层，LSTM专门用于建模时间依赖关系。该编码器由5个Conv2d块组成，旨在从输入特征中提取高级特征或降低分辨率。Conv2d块：一个卷积/反卷积层，然后是批量归一化和激活函数组成。<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2Flocal%2F09UGClb5%2Fitems%2FQNRA3Y3H%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B389.95456960000007%2C735.1752576000011%2C448.72932160000005%2C742.9491264000011%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2Flocal%2F09UGClb5%2Fitems%2FXXHUSN6D%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open/library/items/QNRA3Y3H?page=2">Skip-connection</a></span>：连接编码器和解码器，有利于梯度的流动。

> [!important] 与原始CRN的关键区别
> 原始CRN将实部和虚部作为两个输入通道，仅使用**一个共享的实值卷积核**进行卷积，没有遵循复数乘法规则，网络可能在缺乏先验知识的情况下学习实部和虚部。DCCRN通过复数CNN、复数BN和复数LSTM从**根本上**改进了这一点。

![\<img alt="" width="963" height="324" data-attachment-key="PMRW2UBC" src="attachments/PMRW2UBC.png" ztype="zimage"> | 963](attachments/PMRW2UBC.png)

**架构组成**：
- **编码器**：5个Conv2d块，逐步提取高级语义特征并降低分辨率
- **LSTM层**：2层，建模长时序依赖
- **解码器**：对称于编码器，重建原始尺寸特征
- **Skip-connection**：连接编码器对应层与解码器，改善梯度流动
- **Conv-STFT/iSTFT**：使用STFT核初始化的卷积/反卷积进行波形分析与合成

### 复数模块：编码器与解码器

复数模块通过**模拟复数乘法**来建模幅度和相位之间的相关性。复数编码器块包含：**复数Conv2d + 复数Batch Normalization + 实值PReLU**。

复数Conv2d由**四个传统Conv2d操作**组成，控制整个编码器中的复数信息流：

![\<img alt="" width="944" height="74" data-attachment-key="772ZF48S" src="attachments/772ZF48S.png" ztype="zimage"> | 944](attachments/772ZF48S.png)

![\<img alt="" width="788" height="162" data-attachment-key="T3ZM3ID5" src="attachments/T3ZM3ID5.png" ztype="zimage"> | 788](attachments/T3ZM3ID5.png)

> [!note] 复数卷积数学定义
> 定义复数卷积核 $\mathbf{W} = \mathbf{W}_r + j\mathbf{W}_i$，复数输入矩阵 $\mathbf{X} = \mathbf{X}_r + j\mathbf{X}_i$，则复数卷积输出为：
> $$\mathbf{F}_{out} = (\mathbf{X}_r \circledast \mathbf{W}_r - \mathbf{X}_i \circledast \mathbf{W}_i) + j(\mathbf{X}_r \circledast \mathbf{W}_i + \mathbf{X}_i \circledast \mathbf{W}_r)$$
>
> 其中 $\circledast$ 表示实值卷积操作，$\mathbf{W}_r$ 和 $\mathbf{W}_i$ 分别表示复数卷积核的实部和虚部。

### 复数LSTM

除复数CNN外，DCCRN-CL还引入了**复数LSTM**，同样模拟复数乘法规则。给定复数输入的实部和虚部 $\mathbf{X}_r$ 和 $\mathbf{X}_i$，复数LSTM输出 $\mathbf{F}_{out}$ 定义为：

$$\begin{aligned}
\mathbf{F}_{rr} &= \text{LSTM}_r(\mathbf{X}_r); \quad \mathbf{F}_{ir} = \text{LSTM}_r(\mathbf{X}_i) \\
\mathbf{F}_{ri} &= \text{LSTM}_i(\mathbf{X}_r); \quad \mathbf{F}_{ii} = \text{LSTM}_i(\mathbf{X}_i) \\
\mathbf{F}_{out} &= (\mathbf{F}_{rr} - \mathbf{F}_{ii}) + j(\mathbf{F}_{ri} + \mathbf{F}_{ir})
\end{aligned}$$

其中 $\text{LSTM}_r$ 和 $\text{LSTM}_i$ 分别表示两个传统LSTM（实部和虚部），其权重不共享。DCCRN-CL使用**128单元**（实部和虚部各128，共256等效隐藏单元）。

### 模型配置对比

| 模型 | 编码器通道 | LSTM配置 | Dense层 |
|------|-----------|---------|---------|
| DCCRN-R/C/E | [32, 64, 128, 128, 256, 256] | 2层实值LSTM，256单元 | 1024 |
| DCCRN-CL | [32, 64, 128, 256, 256, 256] | 2层复数LSTM，128单元 (r/i) | 1024 |

卷积核大小：**(5, 2)**（频率×时间），步长：**(2, 1)**。

### 半因果卷积实现

为实现**半因果（semi-causal）**卷积：
- 编码器：在每个Conv2d的时间维度**前端补零**
- 解码器：每个卷积层**超前1帧**
- 最终前瞻量：$6 \times 6.25\text{ms} = 37.5\text{ms}$（符合DNS挑战赛40ms限制）

## 训练目标

### CRM（Complex Ratio Mask）

给定干净语音复数STFT谱 $\mathbf{S}$ 和带噪语音 $\mathbf{Y}$，CRM定义为：

$$\text{CRM} = \frac{\mathbf{Y}_r\mathbf{S}_r + \mathbf{Y}_i\mathbf{S}_i}{\mathbf{Y}_r^2 + \mathbf{Y}_i^2} + j\frac{\mathbf{Y}_r\mathbf{S}_i - \mathbf{Y}_i\mathbf{S}_r}{\mathbf{Y}_r^2 + \mathbf{Y}_i^2}$$

其中 $\mathbf{Y}_r$、$\mathbf{Y}_i$ 和 $\mathbf{S}_r$、$\mathbf{S}_i$ 分别表示带噪和干净复数谱的实部和虚部。

> [!tip] CRM vs SMM
> SMM（Spectral Magnitude Mask）：$\text{SMM} = \frac{|\mathbf{S}|}{|\mathbf{Y}|}$，仅使用幅度信息，忽略相位。
>
> CRM理论上可以**完美重建**干净语音，因为它同时包含幅度和相位信息。CSM（Complex Spectral Mapping）同样具有这一性质。

训练使用**信号逼近（SA）**方法直接最小化掩码处理后的谱与干净谱之间的差异：
- **CSA**（CRM-based SA）：$\mathcal{L}(\tilde{\mathbf{M}} \cdot \mathbf{Y}, \mathbf{S})$
- **MSA**（SMM-based SA）：$\mathcal{L}(|\tilde{\mathbf{M}}| \cdot |\mathbf{Y}|, |\mathbf{S}|)$

### 三种乘法模式

对于DCCRN，可以使用三种乘法模式得到估计的干净语音 $\tilde{\mathbf{S}}$：

![\<img alt="" width="947" height="324" data-attachment-key="B7UGVYHF" src="attachments/B7UGVYHF.png" ztype="zimage"> | 947](attachments/B7UGVYHF.png)

- **DCCRN-C**：以**CSA**的方式得到 $\tilde{\mathbf{S}}$，遵循复数乘法规则：
  $$\tilde{\mathbf{S}} = (\mathbf{Y}_r \cdot \tilde{\mathbf{M}}_r - \mathbf{Y}_i \cdot \tilde{\mathbf{M}}_i) + j(\mathbf{Y}_r \cdot \tilde{\mathbf{M}}_i + \mathbf{Y}_i \cdot \tilde{\mathbf{M}}_r)$$

- **DCCRN-R**：分别估计 $\tilde{\mathbf{Y}}$ 的实部和虚部的掩膜（实值掩码）：
  $$\tilde{\mathbf{S}} = (\mathbf{Y}_r \cdot \tilde{\mathbf{M}}_r) + j(\mathbf{Y}_i \cdot \tilde{\mathbf{M}}_i)$$

- **DCCRN-E**：在**极坐标**下执行，数学上与DCCRN-C类似。不同的是，DCCRN-E使用 **tanh** 激活函数将掩模幅度限制在0到1之间：
  $$\tilde{\mathbf{S}} = \mathbf{Y}_{mag} \cdot \tilde{\mathbf{M}}_{mag} \cdot e^{\mathbf{Y}_{phase} + \tilde{\mathbf{M}}_{phase}}$$

其中极坐标表示为：
$$\begin{aligned}
\tilde{\mathbf{M}}_{mag} &= \sqrt{\tilde{\mathbf{M}}_r^2 + \tilde{\mathbf{M}}_i^2} \\
\tilde{\mathbf{M}}_{phase} &= \arctan 2(\tilde{\mathbf{M}}_i, \tilde{\mathbf{M}}_r)
\end{aligned}$$

## 损失函数

损失函数为 **SI-SNR（Scale-Invariant Source-to-Noise Ratio）**：

![\<img alt="" width="781" height="241" data-attachment-key="XRNEA8XK" src="attachments/XRNEA8XK.png" ztype="zimage"> | 781](attachments/XRNEA8XK.png)

数学定义如下：

$$\begin{cases}
\mathbf{s}_{target} := \frac{\langle \tilde{\mathbf{s}}, \mathbf{s} \rangle \cdot \mathbf{s}}{\|\mathbf{s}\|_2^2} \\
\mathbf{e}_{noise} := \tilde{\mathbf{s}} - \mathbf{s}_{target} \\
\text{SI-SNR} := 10 \log_{10} \left( \frac{\|\mathbf{s}_{target}\|_2^2}{\|\mathbf{e}_{noise}\|_2^2} \right)
\end{cases}$$

其中 $\mathbf{s}$ 和 $\tilde{\mathbf{s}}$ 分别是干净和估计的时域波形，$\langle \cdot, \cdot \rangle$ 表示两个向量之间的点积，$\|\cdot\|_2$ 是欧氏范数（L2范数）。

> [!important] STFT核初始化
> 在送入网络和计算损失函数之前，使用**STFT核初始化的卷积/反卷积模块**来分析/合成波形。这使得卷积编码器/解码器天然具有类似STFT/iSTFT的信号处理能力。

## 实验设置

### 数据集

**WSJ0模拟数据集**：
- 从WSJ0中选取24500条（约50小时）语句，包括131个说话人（男66例，女65例）
- 训练/验证/评估：20000/3000/1500条
- 噪声数据：MUSAN（6.2小时自由声噪声 + 42.6小时音乐），41.8小时用于训练验证，7小时用于评估
- 训练和验证集的语音-噪声混合：随机SNR在 **-5 dB至20 dB** 之间
- 评估集：5个典型SNR（0 dB、5 dB、10 dB、15 dB、20 dB）

**DNS Challenge 2020数据集**（更大规模）：
- 噪声集：180小时，包含150类、65,000个噪声片段
- 干净语音集：超过500小时，来自2150个说话人
- **动态混合**策略：每个epoch随机选择RIR（3000个模拟RIR集，由镜像法生成）进行卷积混响，再以-5至20 dB随机SNR混合
- 10个epoch后模型"看到"的总数据量超过**5000小时**

### 训练设置

| 参数 | 配置 |
|------|------|
| 窗长 / 帧移 | 25 ms / 6.25 ms |
| FFT长度 | 512 |
| 采样率 | 16 kHz |
| 优化器 | Adam |
| 初始学习率 | 0.001（验证loss上升时衰减0.5） |
| 模型选择 | Early stopping |
| 框架 | PyTorch |

### 基线模型

> [!note] LSTM基线
> - **半因果模型**，2层LSTM，每层800单元
> - 时间维度kernel size为7的Conv1d，前瞻6帧实现半因果
> - 输出层：257单元全连接层
> - 输入/输出：带噪/估计干净幅度谱（MSA训练）
> - 参数量：**9.6M**

> [!note] CRN基线
> - **半因果模型**，1个编码器 + 2个解码器（分别处理实部和虚部）
> - Kernel size：(3, 2)，步长：(2, 1)
> - 编码器通道：[16, 32, 64, 128, 256, 256]，输入特征shape：[B, 2, Freq, Time]
> - LSTM隐藏单元：256，Dense：1280
> - 解码器输入通道：[512, 512, 256, 128, 64, 32]
> - 参数量：**6.1M**

> [!note] DCUNET基线
> - DCUNET-16，时间维度步长设为1
> - 编码器通道：[72, 72, 144, 144, 144, 160, 160, 180]
> - 参数量：**3.6M**（但计算量约为DCCRN-CL的**6倍**）

## 实验结果

### WSJ0模拟数据集结果

> [!important] PESQ评分（WSJ0测试集）

| 模型 | 参数量 (M) | 0dB | 5dB | 10dB | 15dB | 20dB | **平均** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Noisy | - | 2.062 | 2.388 | 2.719 | 3.049 | 3.370 | 2.518 |
| LSTM | 9.6 | 2.783 | 3.103 | 3.371 | 3.593 | 3.781 | 3.326 |
| CRN | 6.1 | 2.850 | 3.143 | 3.374 | 3.561 | 3.717 | 3.329 |
| DCCRN-R | 3.7 | 2.832 | 3.192 | 3.488 | 3.717 | 3.891 | 3.424 |
| DCCRN-C | 3.7 | 2.832 | 3.187 | 3.477 | 3.707 | 3.840 | 3.409 |
| **DCCRN-E** | **3.7** | 2.859 | 3.203 | 3.492 | 3.718 | **3.891** | 3.433 |
| **DCCRN-CL** | **3.7** | **2.972** | **3.301** | **3.559** | **3.755** | 3.901 | **3.498** |
| DCUNET | 3.6 | 2.971 | 3.297 | 3.556 | 3.760 | **3.916** | 3.500 |

**分析**：
- 四种DCCRN均**优于**基线LSTM和CRN，验证了复数卷积的有效性
- **DCCRN-CL**（含复数LSTM）全面优于其他DCCRN变体，说明复数LSTM进一步有利于复数目标训练
- DCCRN-CL与DCUNET的PESQ接近（3.498 vs 3.500），但计算量仅为DCUNET的 **1/6**
- 在3.7M参数量下实现了与更大模型LSTM（9.6M）和CRN（6.1M）相比的显著提升

### DNS Challenge 2020结果

> [!important] PESQ评分（DNS Challenge测试集，仅模拟数据）

| 模型 | 参数量 (M) | 前瞻量 (ms) | 无混响 | 有混响 | **平均** |
|------|:---:|:---:|:---:|:---:|:---:|
| Noisy | - | - | 2.454 | 2.752 | 2.603 |
| NSNet (Baseline) | 1.3 | 0 | 2.683 | 2.453 | 2.568 |
| **DCCRN-E [T1]** | **3.7** | 37.5 | **3.266** | 3.077 | **3.171** |
| DCCRN-E-Aug [T2] | 3.7 | 37.5 | 3.209 | **3.219** | **3.214** |
| DCCRN-CL [T2] | 3.7 | 37.5 | 3.262 | 3.101 | 3.181 |
| DCUNET [T2] | 3.6 | 37.5 | 3.223 | 2.796 | 3.001 |

> [!important] MOS主观评分（DNS Challenge盲测集）

| 模型 | 参数量 (M) | 无混响 | 有混响 | 实录 | **平均MOS** |
|------|:---:|:---:|:---:|:---:|:---:|
| Noisy | - | 3.13 | 2.64 | 2.83 | 2.85 |
| NSNet (Baseline) | 1.3 | 3.49 | 2.64 | 3.00 | 3.03 |
| **Track 1** | | | | | |
| **DCCRN-E** | 3.7 | **4.00** | **2.94** | **3.37** | **3.42** 🥇 |
| Team 9 | - | 3.87 | 2.97 | 3.28 | 3.39 |
| Team 17 | - | 3.83 | 3.05 | 3.27 | 3.34 |
| **Track 2** | | | | | |
| Team 9 | - | **4.07** | **3.19** | **3.40** | **3.52** 🥇 |
| **DCCRN-E-Aug** | 3.7 | 3.90 | 2.96 | 3.34 | 3.38 🥈 |
| Team 17 | - | 3.83 | 3.15 | 3.28 | 3.38 |

**分析**：

- **Track 1（实时轨道）**：DCCRN-E以3.7M参数获得**MOS第一名（3.42）**，尤其在无混响场景达到**4.00**分
- **Track 2（非实时轨道）**：DCCRN-E-Aug获得**MOS第二名（3.38）**
- DCCRN-CL虽然在PESQ上略优于DCCRN-E，但内部主观试听发现它在某些片段上可能**过度抑制**语音信号，导致听感不佳——因此在主观评分接近时，主观试听至关重要
- DCUNET在无混响合成集上PESQ不错，但在有混响集上PESQ**大幅下降**（2.796），泛化能力较差
- 为提升混响性能，DCCRN-E-Aug在训练时增加了更多RIR数据

### 推理速度

在 Intel i5-8250U PC 上，DCCRN-E（ONNX导出）的**单帧处理时间为3.12 ms**，满足实时处理需求。

## 结论与展望

**结论**：
1. DCCRN利用**复数网络**对复数谱进行建模，在复数乘法规则约束下，以相似的参数量配置获得了更好的PESQ和MOS表现
2. 结合CED架构和LSTM，以显著降低的可训练参数量和计算成本，有效建模时间上下文
3. 在DNS Challenge上验证了实际竞争力：实时轨道第1，非实时轨道第2

**未来工作**：
- 将DCCRN部署到**边缘设备**等低计算场景
- 提升DCCRN在**混响条件**下的噪声抑制能力

## 相关文献

- [[CRN]] — Tan & Wang, "A convolutional recurrent neural network for real-time speech enhancement," Interspeech 2018
- [[DCUNET]] — Choi et al., "Phase-aware speech enhancement with deep complex u-net," 2019
- [[Deep Complex Networks]] — Trabelsi et al., "Deep complex networks," 2018
- [[CRM]] — Williamson et al., "Complex ratio masking for monaural speech separation," IEEE/ACM TASLP, 2016
- [[DNS Challenge 2020]] — Reddy et al., "The Interspeech 2020 deep noise suppression challenge," 2020

## 思考与启发

- **复数网络优于实值网络处理复数谱**：DCCRN的核心insight是将复数乘法规则显式编码进网络结构，作为"先验知识"约束学习过程
- **轻量化与高性能可以兼得**：3.7M参数 + 1/6 DCUNET计算量，在DNS Challenge上超越多数大模型
- **客观指标与主观听感的差异**：DCCRN-CL在PESQ上更优但主观听感有过度抑制问题，最终选DCCRN-E参赛，说明**主观评估在语音增强中不可或缺**
- **数据增强对泛化至关重要**：动态混合、RIR卷积、宽SNR范围的训练策略显著提升了模型鲁棒性
- **CED + LSTM是高效的架构范式**：相比纯CNN（需大量层数捕捉长程依赖）和纯RNN（缺乏层级特征提取），CED+LSTM在参数量和建模能力间取得了良好平衡
