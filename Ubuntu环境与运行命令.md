# Ubuntu 环境与运行命令

## 版本结论

这个项目大量继承自 TalkNet-ASD。上游 TalkNet-ASD 明确使用 Python 3.7.9，而本项目发布于 2023 年，因此原作者环境最可能是 Python 3.7.9（或非常接近的 Python 3.7 环境）。

不过，当前仓库保留的 `__pycache__/*.cpython-310.pyc` 表明这份修改后的源码已经在 **Python 3.10** 下被导入或运行过。因此本仓库实际建议使用 **Python 3.10.13**，原因是：

- 仍能安装 PyTorch 1.12.1 + CUDA 11.6；
- 能运行代码使用的 PySceneDetect 0.5 旧接口；
- 与当前代码留下的 Python 运行痕迹一致；
- 可以配合 NumPy 1.23.5，保留代码中的 `np.int` 兼容性。

Python 3.8/3.9 也很可能可用，但不建议直接使用 Python 3.11/3.12 或 NumPy 2.x。当前代码把模型和张量直接放到 `.cuda()`，因此必须有 NVIDIA GPU；若要支持 CPU，需要修改源码。

推荐系统为 Ubuntu 22.04 x86_64（系统自带 Python 3.10）。Ubuntu 20.04/24.04 也可使用下面的 Miniconda 环境，但旧版 PyTorch 在 Ubuntu 24.04 上的组合不如 22.04 稳妥。

## 1. 安装 Ubuntu 系统包

```bash
sudo apt update
sudo apt install -y \
  aria2 \
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
- `aria2`：AVA 视频多连接、断点续传下载；没有它时代码会退回 `wget`。
- `tar`：解压 AVA/Columbia 标签。
- `build-essential`：为少数没有预编译 wheel 的 Python 包提供构建工具。

## 2. 检查 NVIDIA 驱动

```bash
nvidia-smi
```

`requirements.txt` 安装的是带 CUDA 11.6 runtime 的 PyTorch wheel，一般不需要另装 CUDA Toolkit 或 cuDNN。CUDA 11.6 对 Linux 驱动的最低兼容版本是 450.80.02，使用 Ubuntu 自动选择的当前受支持驱动更合适。

## 3. 创建 Python 3.10 环境

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

## 4. 安装 Columbia 数据下载工具（可选）

只有运行 `Columbia_test.py --evalCol` 且本地没有 Columbia 视频时才需要。源码调用的命令名是旧的 `youtube-dl`，下面安装维护中的 yt-dlp，并建立兼容命令名：

```bash
sudo curl -L \
  https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux \
  -o /usr/local/bin/yt-dlp
sudo chmod 0755 /usr/local/bin/yt-dlp
sudo ln -sf /usr/local/bin/yt-dlp /usr/local/bin/youtube-dl
```

yt-dlp 需要随视频网站变化更新，所以这里有意使用官方 latest 下载地址，没有把它固定在 Python 依赖文件中。

## 5. 安装检查

```bash
python -c "import torch, torchvision, numpy, scipy, pandas, cv2, scenedetect, python_speech_features; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('numpy:', numpy.__version__)"
ffmpeg -version | head -n 1
aria2c --version | head -n 1
```

预期关键结果为：

- `torch: 1.12.1+cu116`
- `torch CUDA: 11.6`
- `CUDA available: True`
- `numpy: 1.23.5`

如果 `CUDA available` 是 `False`，先处理 NVIDIA 驱动或容器 GPU 映射问题，不要直接开始训练。

## 6. 与当前源码一致的命令

所有命令都应在项目根目录运行。

### 下载并预处理 AVA

```bash
python train.py --dataPathAVA /data/AVADataPath --downloadAVA
```

预处理阶段最高约需 302 GB 空间。删除中间的 `orig_videos` 和 `orig_audios` 后，最终数据小于约 100 GB。

### 训练

最简命令与 README 中的写法等价：

```bash
python train.py --dataPathAVA /data/AVADataPath
```

下面是推荐给普通工作站的显式写法。`--savePath exps/exp1` 是源码默认值，可以省略；`--nDataLoaderThread 8` 不是必需参数，它只是把训练 worker 从默认的 64 降到 8：

```bash
python train.py \
  --dataPathAVA /data/AVADataPath \
  --savePath exps/exp1 \
  --nDataLoaderThread 8
```

`--nDataLoaderThread` 应根据 CPU 核数和内存调整；它只控制训练 DataLoader。当前 `train.py` 的验证/评估 DataLoader 仍把 `num_workers` 硬编码为 64，普通工作站可能需要在源码中把该值调低。

### 使用预训练权重评估 AVA val

最简命令与 README 中的写法等价：

```bash
python train.py --dataPathAVA /data/AVADataPath --evaluation
```

下面只是把 `--savePath exps/exp1` 和 `--evalDataType val` 两个源码默认值显式写出来，运行效果相同，可以省略：

```bash
python train.py \
  --dataPathAVA /data/AVADataPath \
  --savePath exps/exp1 \
  --evalDataType val \
  --evaluation
```

评估不是“只给模型权重就能运行”：当前实现仍会读取 `csv/train_loader.csv`，并需要 val 的 `val_loader.csv`、`val_orig.csv`、`clips_audios/val` 和 `clips_videos/val`。此外，`--nDataLoaderThread` 不会改变评估阶段硬编码的 64 个 worker。

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
