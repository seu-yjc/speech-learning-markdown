---
title: BSRNN - Music Source Separation with Band-split RNN
authors:
  - Yi Luo
  - Jianwei Yu
date: 2022-09-30
tags:
  - music-source-separation
  - band-split
  - RNN
  - frequency-domain
  - semi-supervised
  - MUSDB18
  - deep-learning
  - audio-processing
aliases:
  - Band-split RNN
  - BSRNN
  - 频带分割循环神经网络
cssclasses:
  - paper-note
paper_link: "[[Music Source Separation with Band-split RNN]]"
dataset: MUSDB18-HQ
sample_rate: 44.1k Hz
status: reviewed
---

# BSRNN — Music Source Separation with Band-split RNN

> [!abstract] 核心贡献
> 本文提出了**频带分割循环神经网络（BSRNN）**，一种针对音乐源分离（MSS）的高采样率频率域模型。其核心思路是根据乐器声源特性，将混合频谱显式分割为多个带宽不同的子带，并在子带内与全序列维度交替进行RNN建模，以充分利用音乐信号的频域先验知识。此外，提出了一种**自增强半监督微调流水线**，无需额外训练源活性检测器即可利用无标注数据进一步提升性能。

---

## 一、模型架构总览

BSRNN由 **频带分割模块（Band Split Module）**、**频带与序列建模模块（Band and Sequence Modeling Module）**、**掩码估计模块（Mask Estimation Module）** 三部分构成。

### 1.1 整体流水线

![[attachments/PEL4RAEH.png]]

### 1.2 频带分割模块（Band Split Module）

![[attachments/A5RP94JI.png]]

模块以STFT生成的复值频谱 $\mathbf{X} \in \mathbb{C}^{F \times T}$ 作为输入，将其分割为 $K$ 个子带频谱 $\mathbf{B}_i \in \mathbb{C}^{G_i \times T},\ i=1,\dots,K$，其中预定义的带宽 $\{G_i\}_{i=1}^K$ 满足 $\sum_{i=1}^K G_i = F$。

- 每个子带频谱的实部和虚部被拼接后送入**层归一化（Layer Normalization）**和**全连接层（FC）**，生成实值子带特征 $\mathbf{Z}_i \in \mathbb{R}^{N \times T}$
- 由于各子带带宽 $\{G_i\}$ 可以不同，每个子带都有自己独立的归一化模块和FC层
- 将所有 $K$ 个子带特征 $\{\mathbf{Z}_i\}_{i=1}^K$ 进行融合，生成变换后的全带特征张量 $\mathbf{Z} \in \mathbb{R}^{N \times K \times T}$

> [!tip] 频带分割的核心动机
> 不同于双路径结构中盲目的分块策略，BSRNN **显式地将频率分量拆分为子带特征，并使用对顺序敏感的模块来捕获子带内部的依赖关系**，同时所有子带共享序列建模层以实现并行计算和模型轻量化。

### 1.3 频带与序列建模模块（Band and Sequence Modeling Module）

![[attachments/QLUFZHGI.png]]

BSRNN通过两个不同的残差RNN层进行**交错的序列级（cross-T）和波段级（cross-K）建模**：

- **序列级RNN（RNN across T）**：沿时间维度 $T$ 建模，$K$ 个子带特征共享同一个RNN —— 节省模型大小并允许跨子带并行处理
- **波段级RNN（RNN across K）**：沿频带维度 $K$ 建模，捕捉每个时间帧内跨 $K$ 个子带的频带内特征依赖

每个RNN层的设计相同：
1. **Group Normalization** → 输入归一化
2. **BLSTM** → 双向LSTM（隐藏单元 = $2N = 256$）
3. **FC层** → 线性投影
4. **残差连接** → 输入与FC输出相加

堆叠**12个**此模块（共 **24层** 残差BLSTM），最后一层输出记为 $\mathbf{Q} \in \mathbb{R}^{N \times K \times T}$。

> [!info] 与 Dual-Path RNN 的关键区别
> - BSRNN 用**频带维度（K）** 替代了双路径RNN中的块内/块间维度
> - 实验表明：将序列建模的普通BLSTM替换为双路径RNN **不会提升性能**——双路径结构对BSRNN并非必要
> - 子带拆分引入了**频率顺序敏感性**，与分组拆分模块的"忽略顺序"设计形成对比

### 1.4 掩码估计模块（Mask Estimation Module）

![[attachments/CNSPE5TM.png]]

- $\mathbf{Q}$ 首先被拆分为 $K$ 个子带特征 $\{\mathbf{Q}_i\}_{i=1}^K \in \mathbb{R}^{N \times T}$
- 每个子带特征经过 **层归一化** + **MLP**（单隐藏层，hidden size = $4N = 512$，tanh激活函数）生成复值T-F掩码的实部和虚部 $\mathbf{M}_i \in \mathbb{C}^{G_i \times T}$
- MLP输出层使用 **GLU（Gated Linear Unit）** 作为激活函数
- 所有 $\mathbf{M}_i$ 合并为全带掩码 $\mathbf{M} \in \mathbb{C}^{F \times T}$，与混合频谱 $\mathbf{X}$ 相乘得到目标频谱 $\hat{\mathbf{S}} \in \mathbb{C}^{F \times T}$

> [!note] 使用MLP而非FC的原因
> 遵循 Li & Luo (2022) 的发现：简单的MLP相比普通FC层能更有效地估计T-F掩码。

---

## 二、频带分割方案设计

> [!important] 核心发现
> 低于 1 kHz 的精细分割对捕捉基频（F0）及前几次谐波至关重要 —— 这是BSRNN性能提升的关键因素。

### 2.1 人声（Vocal）分割方案消融实验

| 方案 | 描述 | 子带数 | uSDR (dB) |
|------|------|--------|-----------|
| V1 | 全频带均匀1k Hz带宽 | 22 | 8.15 |
| V2 | <16k: 1k Hz, 16-20k: 2k Hz, 其余1个 | 19 | 8.21 |
| V3 | <8k: 1k Hz, 8-16k: 2k Hz, 16-20k: 1个, 其余1个 | 14 | 8.06 |
| **V4** | **<1k: 100 Hz**, 1-8k: 1k Hz, 8-16k: 2k Hz, 16-20k: 1个, 其余1个 | 23 | **9.51** |
| V5 | <1k: 100 Hz, 1-16k: 1k Hz, 16-20k: 2k Hz, 其余1个 | 28 | 9.57 |
| V6 | <1k: 100 Hz, 1-4k: 500 Hz, 4-8k: 1k Hz, 8-16k: 2k Hz, 16-20k: 1个, 其余1个 | 26 | 9.78 |
| **V7** | **<1k: 100 Hz, 1-4k: 250 Hz, 4-8k: 500 Hz, 8-16k: 1k Hz, 16-20k: 2k Hz, 其余1个** | **41** | **10.04** |

> [!tip] 关键洞察
> - V1→V3 性能持平，**V4 显著提升**（8.06→9.51 dB），主因是低于1k Hz的100 Hz精细分割
> - 1k Hz以下覆盖了人声的基频和前几次谐波，精细分割使波段级RNN能更好地捕捉F0信息
> - 从V4到V7持续提升，证明**精细的频带分割方案对BSRNN至关重要**

### 2.2 各乐器最优分割方案

| 乐器 | 分割方案 | 子带数 |
|------|----------|--------|
| **人声 (Vocal)** | <1k: 100 Hz, 1-4k: 250 Hz, 4-8k: 500 Hz, 8-16k: 1k Hz, 16-20k: 2k Hz, 其余1个 | 41 |
| **贝斯 (Bass)** | <500: 50 Hz, 500-1k: 100 Hz, 1-4k: 500 Hz, 4-8k: 1k Hz, 8-16k: 2k Hz, 其余1个 | 30 |
| **鼓 (Drum)** | <1k: 50 Hz, 1-2k: 100 Hz, 2-4k: 250 Hz, 4-8k: 500 Hz, 8-16k: 1k Hz, 其余1个 | 55 |
| **其他 (Other)** | 与人声V7相同 | 41 |

> [!warning] 频带方案设计原则
> - 不同乐器有不同频率范围、谐波结构和混音技巧 → 需要**各自的频带分割方案**
> - 低频区域（基频所在）需要更精细的分割
> - 高频区域使用粗粒度分割以节省计算量
> - 本文未做穷举网格搜索，**进一步调优频带带宽可能继续提升性能**

---

## 三、训练配置

### 3.1 监督训练

| 参数 | 值 |
|------|-----|
| STFT 窗长 | 2048 |
| STFT 跳步 | 512 |
| 窗函数 | Hanning |
| 特征维度 $N$ | 128 |
| 建模模块数 | 12（共24层BLSTM） |
| BLSTM隐藏单元 | $2N = 256$ |
| MLP隐藏层大小 | $4N = 512$ |
| MLP激活函数 | tanh |
| MLP输出层 | GLU |
| 优化器 | Adam |
| 初始学习率 | $1 \times 10^{-3}$ |
| 学习率衰减 | 每2 epoch × 0.98 |
| 梯度裁剪 | max norm = 5 |
| Batch size | 2（8 GPUs） |
| Epochs | 100（每epoch 10000 batches） |
| Early stopping | 10个连续epoch无最佳验证结果 |

### 3.2 On-the-Fly 数据模拟

1. **显著性片段检测（SAD）**：使用能量阈值法从完整音轨中筛选有效训练段
   - 将音轨分割为6秒长、50%重叠的片段
   - 每片段再分为10个chunk，计算能量
   - 能量阈值 = max(15%能量分位数, 1e-3)
   - 若片段中 >50% 的chunk能量超过阈值 → 保留为有效段
2. **动态混合**：从不同歌曲中随机采样和混合
   - 训练段长度 $T = 3$ 秒
   - 能量随机缩放：$[-10, +10]$ dB
   - 随机丢弃概率：0.1（模拟目标源不活跃的情况）
   - 按最大绝对值归一化到统一尺度

### 3.3 训练目标

$$
\mathcal{L}_{obj} = |\mathbf{S}_r - \hat{\mathbf{S}}_r|_1 + |\mathbf{S}_i - \hat{\mathbf{S}}_i|_1 + |\text{iSTFT}(\mathbf{S}) - \text{iSTFT}(\hat{\mathbf{S}})|_1
$$

结合**频域MAE**（实部+虚部）和**时域MAE**（iSTFT后波形）的联合损失。

> [!note] 训练策略
> 每个目标音轨训练**独立模型**（将MSS视为源提取问题），即4个目标 × 4个独立模型。

---

## 四、半监督微调流水线

![[attachments/IZUI66P5.png]]

### 4.1 核心创新：预训练模型双重角色

> [!important] Self-Boosting Semi-Supervised Finetuning
> 核心理念：将**强预训练模型同时作为伪标签生成器和源活动检测器**，无需额外的外部分类器。

### 4.2 数据采样流程

给定预训练模型 $P$（在有标签数据集 $L$ 上训练）和大规模无标签数据集 $U$：

1. **从 $L$ 采样**：抽取干净目标信号和残余信号（监督训练方式）
2. **从 $U$ 采样**：将混合信号送入 $P$，生成分离的目标信号和残余信号
3. **能量阈值过滤**（$\Delta E > 30$ dB）：
   - 若混合与分离目标之间的能量差 > 30 dB → 混合段被视为**干净残余段**
   - 若混合与分离残余之间的能量差 > 30 dB → 混合段被视为**干净目标段**
4. **伪标签生成**：不满足上述条件的信号作为伪标签使用
5. 采样后的干净信号和伪标签信号混合形成新的训练样本

### 4.3 自增强机制（Self-Boosting）

- 微调阶段定义新模型 $Q$，用 $P$ 初始化
- $P$ 在微调过程中持续被**最新的更优模型**替换
- 当 $Q$ 在验证集上超过 $P$ 时，$P \leftarrow Q$
- 使用 **1750首** 私有歌曲作为无标签数据集 $U$
- 微调初始学习率：$1 \times 10^{-4}$

> [!success] 流水线优势
> - 不需要单独训练源活动检测器或分类器（与 [Demucs](Demucs) 等方法不同）
> - 不增加模型大小（与 RemixIT 不同）
> - 同时使用干净段检测和伪标签生成（**三类现有管道的组合**）
> - 在44.1k Hz全采样率下验证，而非仅限于16k Hz

---

## 五、评估配置

### 5.1 推理流程

- 将完整歌曲分割为 $T = 3$ 秒的chunk，跳步 $P = 0.5$ 秒
- 首尾零填充 $L - P$
- 对所有chunk进行重叠相加（overlap-add）重建完整输出

### 5.2 评估指标

| 指标 | 全称 | 计算方式 | 来源 |
|------|------|----------|------|
| **uSDR** | utterance-level Signal-to-Distortion Ratio | 等同于标准SNR，全曲平均 | MDX Challenge 2021 默认指标 |
| **cSDR** | chunk-level Signal-to-Distortion Ratio | 每1秒chunk的SDR中位数的中位数 | SiSEC 默认指标（bsseval） |

---

## 六、实验结果

### 6.1 评估跳步大小的影响

| 跳步 $P$ (s) | 0.5 | 1.0 | 1.5 | 3.0 |
|--------------|-----|-----|-----|-----|
| uSDR (dB) | 10.04 | 10.00 | 9.94 | 9.75 |

> [!tip] 实践建议
> 选择 $P = 1.5$ 秒可在处理速度和性能之间取得良好平衡——相比 $P = 0.5$ 性能仅下降0.1 dB，但推理速度几乎快3倍。

### 6.2 与 SOTA 系统对比（MUSDB18-HQ & MUSDB18）

| 模型 | Vocal | Bass | Drum | Other | All |
|------|-------|------|------|-------|-----|
| | uSDR / cSDR | uSDR / cSDR | uSDR / cSDR | uSDR / cSDR | uSDR / cSDR |
| ResUNetDecouple+ | – / 8.98 | – / 6.04 | – / 6.62 | – / 5.29 | – / 6.73 |
| CWS-PResUNet | – / 8.92 | – / 5.93 | – / 6.38 | – / 5.84 | – / 6.77 |
| KUIELab-MDX-Net | – / 8.97 | – / 7.83 | – / 7.20 | – / 5.90 | – / 7.47 |
| Hybrid Demucs | – / 8.13 | – / 8.76 | – / 8.24 | – / 5.59 | – / 7.68 |
| **BSRNN** | **10.04 / 10.01** | 6.80 / 7.22 | **8.92 / 9.01** | **6.01 / 6.70** | **7.94 / 8.24** |
| **+ finetuning** | **10.47 / 10.47** | **7.20 / 8.16** | **9.66 / 10.15** | **6.33 / 7.08** | **8.42 / 8.97** |

> [!success] 关键结果
> - BSRNN仅用MUSDB18-HQ有标签数据训练即**在Vocal、Drum、Other上超过所有MDX 2021顶级模型**
> - **人声uSDR达10.04 dB**，显著超越其他系统
> - 贝斯略低于最佳系统（Hybrid Demucs），可能与能量重缩放策略不适配有关（贝斯的能量通常弱于其他乐器）
> - **半监督微调进一步提升所有轨道性能**，贝斯和鼓在cSDR上提升约1 dB

---

## 七、Discussion：与现有方法的联系与区别

### 7.1 与 Dual-Path / Multi-Path 网络的关系

- 形式上相似：都进行交替的局部/全局建模
- **关键区别**：BSRNN的"路径"是**频带维度（K）**，而非时序块维度
- 双路径结构对BSRNN**并非必要**——替换为普通BLSTM不会提升性能
- 双路径最初为小窗长/跳步的时域系统设计，在高分辨率频域建模中优势减弱

### 7.2 与 Group-Splitting 模块的关系

- 分组拆分主要为**轻量化**设计，忽略组内特征的顺序依赖
- 在音乐分离中，不同乐器频率范围和音色差异大 → **显式的、对频率顺序敏感的建模更有效**
- BSRNN共享序列建模层（类似分组通信），但保留了频带内顺序依赖的捕捉能力

### 7.3 与 Super Wideband 模型的关系

- 现有超宽带方法简单地分为低频/高频两段
- BSRNN进行**细粒度的多子带分割**，覆盖更详细的音乐谐波结构
- BSRNN**不做逐频率bin级别建模**（如FullSubNet），以节省计算量、内存和推理速度

---

## 八、关键超参数总结

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 采样率 | 44.1k Hz | 音乐信号高标准采样率 |
| STFT窗长 | 2048 | ~46ms @ 44.1k Hz |
| STFT跳步 | 512 | 75% overlap |
| 频率bin数 $F$ | 1025 | 2048/2 + 1 |
| 特征维度 $N$ | 128 | 所有子带统一维度 |
| RNN层数 | 24 | 12个模块 × 2层BLSTM |
| 总参数量 | ~ | 每个目标独立模型 |
| 训练段长 $T$ | 3 秒 | On-the-fly混合 |
| SAD段长 $L$ | 6 秒 | 显著性检测用 |

---

## 九、未来工作方向

- 研究更智能的方式将**源的先验知识**融入频带分割方案（替代大范围网格搜索）
- 在更多乐器类型和**通用音频提取/分离任务**上验证
- 扩大半监督无标签数据集规模（本文仅1750首）
- 引入**有损压缩格式（如mp3）** 的歌曲，让波段级RNN学习处理高频失真
- 探索额外数据量和数据类型对监督训练和半监督微调的影响

---

## 十、相关论文连接

- [[gtcrn]] — 语音增强超低计算量模型，与BSRNN同为频域掩码估计范式
- [[dccrn]] — 深度复数卷积循环网络，频域相位感知建模的代表作
- [[Semi-Supervised Speech Enhancement Based On Speech Purity]] — 半监督语音增强的另一种思路
- [[Learning-Based Multi-Channel Speech Presence Probability Estimation using A Low-Parameter Model and Integration with MVD]] — 多通道语音存在概率估计

---

> [!quote] 论文信息
> Yi Luo, Jianwei Yu. "Music Source Separation with Band-split RNN." arXiv:2209.15174, Sep 2022.
> 发表于 MDX Challenge 2021 后的顶级MSS方法。
