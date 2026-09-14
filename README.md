# Light-ASD-QMF

本项目地址：[Gz171189/Light-ASD-QMF](https://github.com/Gz171189/Light-ASD-QMF)。本项目选用 [Junhua-Liao/Light-ASD](https://github.com/Junhua-Liao/Light-ASD)（CVPR 2023）作为 baseline，在此基础上研究逐帧可靠性融合，并完成 WASD 训练与评估适配。

本文按当前源码中的运行入口、参数、数据路径和保存逻辑整理；依赖版本以 `requirements.txt` 为准，实验成绩以各轮日志中的实际 mAP 为准。下列命令使用 Bash，均在项目根目录运行；数据路径需替换为实际路径。

## 已有实验与项目结构

| 实验目录 | 实验含义 | 日志最佳 AVA val mAP |
| --- | --- | --- |
| `exps/exp1` | 本项目复现的原版 Light-ASD baseline 结果 | 93.75%，第 27 轮 |
| `exps/qmf_sync_rank_seed0` | M03（`qmf_sync_rank`）、随机种子 0，在 AVA 上训练的结果 | 93.90%，第 21 轮 |

最佳成绩取完整日志中各轮实际 `mAP` 的最大值，对应 baseline 的 `model_0027.model` 与 M03 的 `model_0021.model`。baseline 第 31 轮的 93.26% 只是该轮结果，不能替代第 27 轮的历史最佳。93.90% 属于 AVA 结果；WASD 的分数需要读取单独运行 WASD 评估后生成的评估日志。本次文档整理没有重新训练或评测。

| 文件或目录 | 用途 |
| --- | --- |
| `train.py` | AVA 下载、预处理、训练和评估 |
| `WASD_train.py` | WASD train 训练与 val 验证 |
| `WASD_test.py` | 指定模型在 WASD val 上独立推理与官方评估 |
| `ASD.py`、`model/`、`loss.py` | 模型封装、编码器、融合模块、GRU 与损失 |
| `dataLoader.py`、`utils/` | 数据加载、预处理与运行辅助逻辑 |
| `Columbia_test.py` | Columbia 评估和本地视频演示 |
| `weight/` | 上游提供的 AVA 预训练与 TalkSet 微调权重 |
| `exps/` | 本地实验权重、日志和预测结果 |
| `requirements.txt` | 运行所需的 Python 依赖，独立保留 |

## 融合模式

| `--fusionMode` | 含义 |
| --- | --- |
| `sum` | 原版 baseline：`audio + visual -> GRU` |
| `qmf` | M01：两个独立轻量 MLP 根据 `[B,T,128]` 音视频嵌入预测逐帧可靠性 |
| `qmf_sync` | M02：视觉辅助分类 logits 的 energy 与候选人条件化 AV 同步分数共同决定融合权重 |
| `qmf_sync_rank` | M03：视觉质量头接收中心化 visual-logit energy 和视觉嵌入，并使用逐帧视觉损失的排序监督 |

可靠性头末层采用零初始化，使初始融合保持原版相加行为。M02/M03 的融合权重满足 `WA + WV = 2`，初始化时均为 1。M03 使用以下损失，不引入纯音频候选人分类损失 `L_A` 或 `L_balance`：

```text
L = L_AV + 0.5 * L_V + lambdaSync * L_sync + lambdaRank * L_rank,V
```

当前 AVA 入口默认使用 `qmf`；运行原版 baseline 时显式指定 `--fusionMode sum`。WASD 从头训练默认使用 `qmf_sync_rank`，独立评估默认自动识别权重架构。

2026-09-14 起，共用音频混合增强采用服务器版本的数值修复：先将 PCM 转为 `float64` 计算功率，再将混合结果裁剪至 `int16` 范围，避免平方溢出和转换越界。此修复对 AVA/WASD 的所有训练模式生效，保留原有三张量数据加载接口；验证和测试不启用该增强。已有 93.90% 等历史成绩不代表修复后的重新训练结果；旧 checkpoint 仍可加载，但续训会使用修复后的增强，实验记录应注明这一变化。

## Ubuntu 环境安装

### 版本结论

安装示例沿用 Ubuntu 22.04 x86_64 和 Python 3.10 环境。当前 `requirements.txt` 固定了 PyTorch `1.12.1+cu116`、torchvision `0.13.1+cu116`、NumPy `1.23.5` 和 PySceneDetect `0.5.6.1`。

`model/faceDetector/s3fd/box_utils.py` 使用 `np.int`，`Columbia_test.py` 使用旧版 PySceneDetect `VideoManager` 接口，因此安装时保持依赖清单中的版本。当前模型封装和训练/推理路径直接调用 `.cuda()`，正式运行需要 NVIDIA GPU。

### 1. 安装 Ubuntu 系统包

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  tar \
  wget
```

这些程序的用途如下：

- `ffmpeg`：抽取音频、视频转码、切帧和合并结果；必须安装真实的命令行程序，不是同名 Python 包。
- `wget`：当前 AVA 下载函数调用的下载工具。
- `tar`：解压 AVA/Columbia 标签。
- `build-essential`：为少数没有预编译 wheel 的 Python 包提供构建工具。

### 2. 检查 NVIDIA 驱动

```bash
nvidia-smi
```

按 `requirements.txt` 的环境说明，PyTorch wheel 已包含 CUDA 11.6 运行库，系统需要兼容的 NVIDIA 驱动，一般不需要另外安装 CUDA Toolkit 或 cuDNN。

### 3. 创建 Python 3.10 环境

系统自带 Python 版本会随 Ubuntu 版本变化，使用独立环境更容易复现：

```bash
conda create -y -n Light-ASD python=3.10 pip
conda activate Light-ASD
```

进入项目目录并安装 Python 依赖：

```bash
cd /path/to/Light-ASD-main
python -m pip install -r requirements.txt
```

### 4. 安装 Columbia 数据下载工具（可选）

只有运行 `Columbia_test.py --evalCol` 且本地没有 Columbia 视频时才需要。源码调用的命令名是旧的 `youtube-dl`，下面安装维护中的 yt-dlp，并建立兼容命令名：

```bash
sudo curl -L \
  https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux \
  -o /usr/local/bin/yt-dlp
sudo chmod 0755 /usr/local/bin/yt-dlp
sudo ln -sf /usr/local/bin/yt-dlp /usr/local/bin/youtube-dl
```

yt-dlp 需要随视频网站变化更新，所以这里有意使用官方 latest 下载地址，没有把它固定在 Python 依赖文件中。

### 5. 安装检查

```bash
python -c "import torch, torchvision, numpy, scipy, pandas, cv2, scenedetect, python_speech_features; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('numpy:', numpy.__version__)"
ffmpeg -version | head -n 1
```

预期关键结果为：

- `torch: 1.12.1+cu116`
- `torch CUDA: 11.6`
- `CUDA available: True`
- `numpy: 1.23.5`

如果 `CUDA available` 是 `False`，先处理 NVIDIA 驱动或容器 GPU 映射问题，不要直接开始训练。

## AVA 数据准备、训练与评估

### 下载和预处理

```bash
python train.py --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker --downloadAVA
```

正式参数名为 `--downloadAVA`。预处理阶段最高约需 302 GB 空间；完成切片后删除中间的 `orig_videos` 和 `orig_audios`，最终数据小于约 100 GB。

### 训练

以下命令用于启动新实验，输出到新的目录。已有 `exps/exp1` 和 `exps/qmf_sync_rank_seed0` 作为历史实验保留。

重新训练原版固定求和 baseline：

```bash
python train.py \
  --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker \
  --savePath exps/baseline_new_seed0 \
  --fusionMode sum --seed 0 --maxEpoch 30
```

训练 M03，随机种子设为 0：

```bash
python train.py \
  --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker \
  --savePath exps/qmf_sync_rank_new_seed0 \
  --fusionMode qmf_sync_rank --seed 0 --maxEpoch 30
```

M01/M02 消融分别使用 `--fusionMode qmf` / `--fusionMode qmf_sync`，并更换实验输出目录。需要先试跑时，可将 `--maxEpoch 30` 改为 `--maxEpoch 1`。

默认参数为 `lr=0.001`、`lrDecay=0.95`、`batchSize=2000`（动态批次帧数预算）、`testInterval=1`；`lambdaSync=0.1`、`lambdaRank=0.1`、`rankMargin=0.1`、`rankMinLossGap=0.05`，两个 temperature 均为 `1.0`。

### 输出和续训

相对于 `savePath`，主要输出为：

| 路径 | 内容 |
| --- | --- |
| `score.txt` | 训练与验证日志 |
| `model/model_XXXX.model` | 纯模型参数，用于 AVA 评估 |
| `model/training_XXXX.checkpoint` | 完整训练状态，用于续训 |
| `val_res.csv` | 最近一次 AVA val 预测结果 |

AVA 按 `testInterval` 验证和保存，默认每轮一次。日志包含 mAP、可靠性的 mean/std/min/max、显存峰值和同步/排序损失；M03 还记录 `VScoreLossCorr`。

使用相同 `savePath` 会自动续训：优先读取最新的完整训练 checkpoint，否则退回最新 `.model`，后者无法恢复 Adam 状态。续训需保持原模型和训练参数，`maxEpoch` 表示最终总轮数。`train.py` 的 `--pretrainModel` 仅在 `--evaluation` 时使用，不用于训练初始化。

### AVA val 评估

评估上游提供的 baseline 权重：

```bash
python train.py \
  --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker \
  --savePath exps/ava_upstream_eval \
  --evaluation --evalDataType val --fusionMode sum \
  --pretrainModel weight/pretrain_AVA_CVPR.model
```

评估本地 M03 第 21 轮模型：

```bash
python train.py \
  --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker \
  --savePath exps/ava_m03_epoch21_eval \
  --evaluation --evalDataType val --fusionMode qmf_sync_rank \
  --pretrainModel exps/qmf_sync_rank_seed0/model/model_0021.model
```

评估自己复现的 baseline 最佳模型（第 27 轮）：

```bash
python train.py \
  --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker \
  --savePath exps/ava_baseline_epoch27_eval \
  --evaluation --evalDataType val --fusionMode sum \
  --pretrainModel exps/exp1/model/model_0027.model
```

`weight/pretrain_AVA_CVPR.model` 是上游权重，与自己训练的 baseline 权重区分使用。

模型的融合模式和构造参数应与训练时一致；如果当时采用非默认 hidden dim、dropout、minReliability 或 temperature，应补充相应参数。

即使只评估，当前 AVA 入口仍会读取 `csv/train_loader.csv`，并需要 val 的 `val_loader.csv`、`val_orig.csv`、`clips_audios/val` 和 `clips_videos/val`。

### Checkpoint 配置简述

原 `CHECKPOINT_CONFIG.md` 涉及的配置恢复逻辑仍在 `ASD.py` 和 `utils/checkpoint_config.py` 中；删除说明文件不影响功能。完整 `.checkpoint` 除训练状态外，还保存六项模型构造配置：融合模式、可靠性下限、energy/fusion temperature、可靠性头 hidden dim 和 dropout。

WASD 入口加载新 checkpoint 时恢复这些配置，显式参数与已保存配置或权重架构冲突时会报错。旧 `.model` 可推断架构和 QMF hidden dim，但无法反推缺失的标量；缺失值采用命令行参数或兼容默认值并提示。AVA 评估仍使用纯 `.model` 和命令行构造参数，不会自动恢复这些配置。WASD 续训还会读取其完整 checkpoint 中保存的训练设置。

原说明还涉及 AVA 预处理错误检查：当前代码检查输入、裁剪及写入失败，并先写临时文件再发布有效输出；已有非空目标会跳过。

## WASD 独立评估与训练

这里的“测试”指 WASD val 上的独立评估，不是额外的 held-out test 集。使用 AVA 模型直接评估 WASD，与在 WASD 上训练后验证，是两种不同实验。

### 数据准备

WASD 需事先准备好数据；AVA 下载命令不能用于下载 WASD。独立评估需要：

```text
/root/autodl-tmp/WASD/
├── csv/val_loader.csv
├── csv/val_orig.csv
├── clips_audios/val/
├── clips_videos/val/
└── eval/
    ├── WASD_evaluation.py
    └── dataset_division.txt
```

WASD 训练还需要 `csv/train_loader.csv`、`csv/train_orig.csv`、`clips_audios/train/` 和 `clips_videos/train/`。模型只在 train 上更新参数，val 用于验证和模型选择。

WASD 训练、续训和独立评估均要求 `savePath` 尚不存在，空目录也不允许；不要提前创建该输出目录，重复运行请换新目录名。

### 用 AVA 第 21 轮 M03 模型评估 WASD

以下整合原 `WASD测试命令.txt` 的用途，模型路径是本地已有的纯权重文件：

```bash
python WASD_test.py \
  --dataPathWASD /root/autodl-tmp/WASD \
  --pretrainModel exps/qmf_sync_rank_seed0/model/model_0021.model \
  --savePath exps/qmf_sync_rank_wasd_epoch21 \
  --wasdEvalDir /root/autodl-tmp/WASD/eval
```

该命令只在 WASD val 上推理和评估，不训练或更新模型。输出为指定目录下的 `val_res.csv` 和 `wasd_eval.txt`，WASD 指标以此日志为准。省略 `--wasdEvalDir` 时仅生成预测，不计算官方指标。

旧 `.model` 可从权重识别架构，但不包含全部训练设置；如果训练采用了非默认构造标量，需显式传入实际值。

### 在 WASD 上训练

从头训练 M03：

```bash
python WASD_train.py \
  --dataPathWASD /root/autodl-tmp/WASD \
  --wasdEvalDir /root/autodl-tmp/WASD/eval \
  --savePath exps/wasd_m03_seed0 \
  --fusionMode qmf_sync_rank --maxEpoch 30 --seed 0
```

使用架构兼容的 AVA M03 权重初始化，优化器从头开始：

```bash
python WASD_train.py \
  --dataPathWASD /root/autodl-tmp/WASD \
  --wasdEvalDir /root/autodl-tmp/WASD/eval \
  --savePath exps/wasd_m03_from_ava \
  --pretrainModel exps/qmf_sync_rank_seed0/model/model_0021.model \
  --fusionMode qmf_sync_rank --maxEpoch 30 --seed 0
```

从上述 WASD 实验已生成的完整训练 checkpoint 继续，输出到另一个新目录：

```bash
python WASD_train.py \
  --dataPathWASD /root/autodl-tmp/WASD \
  --wasdEvalDir /root/autodl-tmp/WASD/eval \
  --savePath exps/wasd_m03_resume \
  --resume exps/wasd_m03_from_ava/model/training_0001.checkpoint \
  --maxEpoch 30
```

`--resume` 与 `--pretrainModel` 互斥；前者恢复训练状态，后者只初始化权重。`maxEpoch=30` 表示训练到第 30 轮。上述续训文件须先由训练生成，不能把 `.model` 直接改扩展名作为完整训练状态。

每轮保存纯权重与完整训练 checkpoint；按 `testInterval` 以及最终轮进行验证，保存 `val_XXXX/val_res.csv`、`val_XXXX/wasd_eval.txt`，并更新最优 `model/best.checkpoint`。实验目录还记录 `score.txt` 和 `run_config.json`。

## Columbia 评估与本地视频演示

### Columbia 数据集评估

```bash
python Columbia_test.py \
  --evalCol \
  --colSavePath /data/colDataPath \
  --pretrainModel weight/pretrain_AVA_CVPR.model
```

### 本地视频演示

例如输入文件为 `/data/demo/0001.mp4`：

```bash
python Columbia_test.py \
  --videoName 0001 \
  --videoFolder /data/demo \
  --pretrainModel weight/pretrain_AVA_CVPR.model
```

输出位于 `/data/demo/0001/pyavi/video_out.avi`。注意该脚本每次运行都会删除并重建同名视频的中间输出目录。

需要使用上游 TalkSet 微调权重时，将以上命令的 `--pretrainModel` 替换为 `weight/finetuning_TalkSet.model`。

## 来源与引用

本项目基于 Light-ASD 改进，保留原 `LICENSE`。使用原版代码或模型权重时，请引用 baseline 论文：

```bibtex
@InProceedings{Liao_2023_CVPR,
    author    = {Liao, Junhua and Duan, Haihan and Feng, Kanghui and Zhao, Wanbing and Yang, Yanbing and Chen, Liangyin},
    title     = {A Light Weight Model for Active Speaker Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22932-22941}
}
```

代码继承与致谢：[TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD)。
