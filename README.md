## 轻量级主动说话人检测模型
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-light-weight-model-for-active-speaker/audio-visual-active-speaker-detection-on-ava)](https://paperswithcode.com/sota/audio-visual-active-speaker-detection-on-ava?p=a-light-weight-model-for-active-speaker)

本仓库包含我们 [论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Liao_A_Light_Weight_Model_for_Active_Speaker_Detection_CVPR_2023_paper.pdf)（CVPR 2023）的代码和模型权重：

> A Light Weight Model for Active Speaker Detection
> Junhua Liao, Haihan Duan, Kanghui Feng, Wanbing Zhao, Yanbing Yang, Liangyin Chen

扩展版本（[LR-ASD: 轻量鲁棒的主动说话人检测网络](https://junhua-liao.github.io/Junhua-Liao/publications/papers/IJCV_2025.pdf)，IJCV 2025，[代码](https://github.com/Junhua-Liao/LR-ASD)）


***
### 在 AVA-ActiveSpeaker 数据集上评估

#### 数据准备
使用以下代码下载并预处理 AVA 数据集：
```
python train.py --dataPathAVA AVADataPath --download
```
AVA 数据集及标签将下载至 `AVADataPath`。

#### 训练
可使用以下命令在 AVA 数据集上训练模型：
```
python train.py --dataPathAVA AVADataPath
```
`exps/exps1/score.txt`：输出分数文件，`exps/exp1/model/model_00xx.model`：训练好的模型，`exps/exps1/val_res.csv`：验证集预测结果。

当前版本默认使用 QMF-inspired 的逐帧可靠性融合。两个轻量 MLP 分别从
`[B,T,128]` 的音频、视觉嵌入预测逐帧可靠性，再替换原始固定相加。
可靠性头采用零初始化，因此加载原始 Light-ASD 权重时，初始融合严格等价于
`audio + visual`。使用 `--fusionMode sum` 可运行原始固定相加基线。
正式训练时请使用新的实验目录，避免续训 `exps/exp1` 中已有的基线模型：
```
python train.py --dataPathAVA AVADataPath --fusionMode qmf --savePath exps/qmf_framewise_mvp
```

研究模式：

- `sum`：原始 Light-ASD 固定求和；
- `qmf`：M01，两个独立 MLP 门控，保留作消融；
- `qmf_sync`：M02，面向 ASD 调整的非对称 QMF。视觉权重使用现有视觉
  辅助分类 logits 的 energy，音频侧使用候选人条件化的音视频对应分数，
  不增加无效的纯音频候选人分类损失；
- `qmf_sync_rank`：M03。在 M02 的候选人条件化 AV 同步分支上，将中心化
  visual-logit energy 与视觉 embedding 输入零初始化的视觉质量头，并使用
  视觉逐帧损失排序监督质量方向。不使用 `L_A`，也不使用 `L_balance`。

M02 的权重始终满足 `WA+WV=2`，初始化时 `WA=WV=1`，所以严格等价于
原始 `audio+visual`。在 AVA 上启动独立实验：

```bash
python train.py --dataPathAVA AVADataPath \
  --fusionMode qmf_sync --lambdaSync 0.1 \
  --energyTemperature 1.0 --fusionTemperature 1.0 --seed 0 \
  --savePath exps/qmf_sync_seed0
```

M03 使用
`L=L_AV+0.5*L_V+lambdaSync*L_sync+lambdaRank*L_rank,V`。默认配置的
1-epoch 服务器冒烟实验（先创建目录，确保 `tee` 可以打开日志）：

```bash
mkdir -p exps/qmf_sync_rank_smoke_seed0
python train.py \
  --dataPathAVA /root/autodl-tmp/AVA-ActiveSpeaker \
  --fusionMode qmf_sync_rank \
  --maxEpoch 1 \
  --savePath exps/qmf_sync_rank_smoke_seed0 \
  2>&1 | tee exps/qmf_sync_rank_smoke_seed0/console.log
```

`lambdaSync=0.1`、`lambdaRank=0.1`、`rankMargin=0.1`、
`rankMinLossGap=0.05`、两个 temperature 均为 `1.0`、`seed=0`，以上均为
默认值。`nDataLoaderThread` 未指定时使用默认值 `64`。M03 日志额外记录
`LossRank` 和训练/验证 `VScoreLossCorr`；后者应逐渐为负，表示视觉帧损失
越高，预测的视觉质量分数越低。

每轮同时保存 `model_XXXX.model`（仅参数，用于评测）和
`training_XXXX.checkpoint`（模型、Adam、scheduler、epoch、全局 best 和
随机状态，用于续训）。相同 `savePath` 会优先从完整 checkpoint 的下一轮继续；
`maxEpoch` 表示最终总轮数。

无需数据集即可运行轻量检查：
```
python sanity_check_qmf.py
```

每个训练和验证 epoch 结束时会输出音频、视觉可靠性的
`mean/std/min/max`，并记录训练和验证阶段的峰值显存。上述信息也会写入
实验目录下的 `score.txt`。可指定任意 checkpoint 进行验证：
```
python train.py --evaluation --fusionMode qmf \
  --pretrainModel exps/qmf_framewise_mvp/model/model_0001.model \
  --dataPathAVA AVADataPath --savePath exps/qmf_framewise_mvp_eval
```

#### 测试
模型权重已放置在 `weight` 文件夹中，在验证集上的表现为 `mAP: 94.06%`。可使用以下命令进行验证：
```
python train.py --dataPathAVA AVADataPath --evaluation
```


***
### 在 Columbia ASD 数据集上评估

#### 测试
在 AVA 数据集上训练的模型权重已放置在 `weight` 文件夹中，运行以下代码：
```
python Columbia_test.py --evalCol --colSavePath colDataPath
```
Columbia ASD 数据集及标签将下载至 `colDataPath`，可得到如下 F1 结果：
| 名称 |  Bell  |  Boll  |  Lieb  |  Long  |  Sick  |  平均  |
|----- | ------ | ------ | ------ | ------ | ------ | ------ |
|  F1  |  82.7% |  75.7% |  87.0% |  74.5% |  85.4% |  81.1% |

我们还提供了在 TalkSet 数据集上微调的模型权重（论文中因篇幅限制未展示）。运行以下代码：
```
python Columbia_test.py --evalCol --pretrainModel weight/finetuning_TalkSet.model --colSavePath colDataPath
```
可得到如下 F1 结果：
| 名称 |  Bell  |  Boll  |  Lieb  |  Long  |  Sick  |  平均  |
|----- | ------ | ------ | ------ | ------ | ------ | ------ |
|  F1  |  97.7% |  86.3% |  98.2% |  99.0% |  96.3% |  95.5% |


***
### 使用预训练 Light-ASD 模型的演示

将原始视频（支持 `.mp4` 和 `.avi`）放入 `demo` 文件夹，例如 `0001.mp4`：
```
python Columbia_test.py --videoName 0001 --videoFolder demo
```
默认加载在 AVA-ActiveSpeaker 数据集上训练的权重。若要加载在 TalkSet 上微调的权重，执行：
```
python Columbia_test.py --videoName 0001 --videoFolder demo --pretrainModel weight/finetuning_TalkSet.model
```
输出视频为 `demo/0001/pyavi/video_out.avi`，其中主动说话人用绿框标注，非主动说话人用红框标注。


***
### 引用

如果您使用了本代码或模型权重，请引用我们的论文：

```
@InProceedings{Liao_2023_CVPR,
    author    = {Liao, Junhua and Duan, Haihan and Feng, Kanghui and Zhao, Wanbing and Yang, Yanbing and Chen, Liangyin},
    title     = {A Light Weight Model for Active Speaker Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22932-22941}
}
```
```
@article{Liao_2025_IJCV,
  title     = {LR-ASD: Lightweight and Robust Network for Active Speaker Detection},
  author    = {Liao, Junhua and Duan, Haihan and Feng, Kanghui and Zhao, Wanbing and Yang, Yanbing and Chen, Liangyin and Chen, Yanru},
  journal   = {International Journal of Computer Vision},
  pages     = {1--21},
  year      = {2025},
  publisher = {Springer}
}
```


***
### 致谢
感谢 TaoRuijie 的开源[仓库](https://github.com/TaoRuijie/TalkNet-ASD)对本研究的支持。
