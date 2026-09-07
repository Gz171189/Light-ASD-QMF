# Checkpoint 配置恢复与 AVA 预处理修复

本次仅修复推理配置可复现性、WASD 安全恢复和 AVA 预处理错误处理。

## 文件改动

| 文件 | 修改内容 |
| --- | --- |
| `ASD.py` | 将原有六个构造参数记录为普通 Python `model_config` 字典；完整 `saveCheckpoint()` 保存该字典。保留顶层 `fusion_mode` 和纯权重 `saveParameters()` 格式。 |
| `utils/checkpoint_config.py` | 集中定义配置字段、默认值、旧权重架构推断、metadata 与权重结构交叉检查、显式 CLI 冲突拒绝和配置来源日志。 |
| `WASD_test.py` | 在构造模型前恢复配置；使用 `None` 区分省略与显式参数；增加 hidden dim/dropout 参数；保留严格权重检查及原推理、预处理、评测路径。 |
| `utils/tools.py` | 缺失音频/视频、读帧失败、无效 crop、截取越界、写入失败立即报错；检查 ffmpeg 返回码；临时写入成功后发布目标文件；释放 VideoCapture；统计 processed/skipped_existing/failed。音频缓存同时记录对应采样率。 |
| `sanity_check_checkpoint_config.py` | 无数据集 CPU 自检，覆盖真实保存/加载 API、配置冲突、四种旧架构、参数名/shape、sum 等价及异常 metadata。 |
| `sanity_check_preprocessing.py` | 用临时音频、视频和模拟失败验证 fail fast、输出缺失、完整及部分断点续跑。 |
| `CHECKPOINT_CONFIG.md` | 本说明及验证报告。 |

新 checkpoint 的 `model_config` 只含：

```python
{
    'fusion_mode': 'qmf_sync_rank',
    'min_reliability': 0.2,
    'energy_temperature': 0.8,
    'fusion_temperature': 0.5,
    'reliability_hidden_dim': 48,
    'reliability_dropout': 0.15,
}
```

这是当前 `ASD_Model` 的全部可配置构造参数。feature dim 等常量仍由原模型代码定义。
没有加入学习率、Loss 权重、rank margin 等纯训练参数，也没有新增参数张量或网络层。

## 新旧 checkpoint 的处理

| 类型 | WASD 处理 |
| --- | --- |
| 旧 `model_XXXX.model` | 从 tensor 名称识别 sum/qmf/qmf_sync/qmf_sync_rank，从 shape 识别 QMF hidden dim；其余参数使用 CLI/default 并明确打印 warning。sum 没有可靠性头，hidden dim 不可推断且不参与计算。 |
| 旧 `training_XXXX.checkpoint` | 读取已有顶层 `fusion_mode`（以及存在的六项配置字段），核对 tensor 架构；缺失字段使用兼容值并打印 warning。不会仅因缺少新 metadata 而报错。 |
| 新 `training_XXXX.checkpoint` | 自动读取 `model_config` 六项配置并构造模型；显式 CLI 值一致则允许，不一致则 `ValueError`；metadata 与 tensor 架构矛盾也拒绝。 |

旧权重缺失标量的默认值为 minReliability=0.1、energyTemperature=1.0、
fusionTemperature=1.0、reliabilityDropout=0.1；日志明确标注这些值来自 CLI/default，
不会声称它们已从旧权重恢复。无法从旧权重反推训练时真实标量。

AVA 的 `loadParameters()`、训练默认参数、训练循环、续训入口均未修改。
旧 `.model` 继续按已有 AVA 调用方式使用；自动恢复配置是本次 WASD 入口新增的行为。
完整 checkpoint 包含 Python/NumPy RNG 状态，WASD 显式使用 `weights_only=False`
读取用户自己的可信模型，以兼容新版本 PyTorch 的加载默认值变化。

## 推荐 WASD 命令

对修改后新保存的完整 checkpoint，无需重复指定训练时的六个 QMF 参数：

```bash
python WASD_test.py \
  --dataPathWASD /root/WASD \
  --pretrainModel exps/qmf_sync_rank_seed0/model/training_0021.checkpoint \
  --savePath exps/wasd_qmf_sync_rank \
  --wasdEvalDir /root/WASD/eval
```

同名旧 checkpoint 仍会走缺失配置兼容分支；升级代码无法补回原文件中不存在的信息。
例如旧模型的训练参数不是默认值时，需要继续手动传入对应标量。

## AVA 预处理语义

成功存在的非空目标文件保留 skip；空文件和无效目标直接报错。
只有需要生成目标时才访问原始输入，因此已完成的音频片段或视频帧可在原始输入
已移走的情况下继续被跳过。图片支持按帧跳过部分完成的 entity。

错误包含 split、video_id、entity_id、timestamp 和源/目标路径。
整段原始音频提取不对应特定 entity/timestamp，使用 `N/A`。
视频/音频切片按原有时间索引与 crop 规则生成；未改 MFCC、灰度化、112×112 resize、normalization。
新输出先写入同一目录下的临时文件，写入成功且目标非空后 `os.replace`；
失败的输出不会进入“已完成”文件集合。失败在发生点 raise，只有正常完成 split 才打印
`failed=0`。音频计数单位是 entity，视频计数单位是帧。

检查保持轻量：对已存在文件检查文件类型及非空，不重新解码全部历史文件，
不保证识别历史运行遗留的所有非空损坏文件。

## 数学行为与实际验证

与修改前快照逐字节比较：`model/Model.py`、`model/Encoder.py`、
`model/Classifier.py`、`loss.py`、`dataLoader.py`、`train.py` 均未修改；
`model/faceDetector` 未编辑。ASD 训练/验证函数和 WASD forward、infer、预测导出、
官方 evaluator 适配函数的 AST 均保持相同。

已对照官方 [Light-ASD tools.py](https://github.com/Junhua-Liao/Light-ASD/blob/main/utils/tools.py)：
官方没有当前项目的断点跳过逻辑，也没有完整的读帧/写文件检查。
本次保留本项目已有缓存和 crop 边界裁剪，只把失败分支改为明确错误并验证输出。
Encoder、Classifier 与官方源码一致；同权重随机输入下，lossAV/lossV
在 r=1.0 与 r=1.28 时和官方输出逐值相等。

以下定义保持原样：四种 fusion 公式、可靠性头、GRU、
`L_AV + 0.5 * L_V + lambdaSync * L_sync + lambdaRank * L_rank`。
sum 路径仍是 `audio + visual -> GRU`，自检以 `torch.allclose(rtol=0, atol=0)` 通过。

运行：

```bash
python sanity_check_checkpoint_config.py
python sanity_check_preprocessing.py
```

实际验证环境：Windows、Python 3.10.20、PyTorch 2.5.1+cu121，CPU 自检；
NumPy 2.2.6、OpenCV 4.13.0。未修改项目的 `requirements.txt`。

- checkpoint：6 项测试通过，包括非零可靠性头权重下恢复前后推理输出逐值相等。
- preprocessing：11 项测试通过，包括合成媒体生成、缺输入、打不开/无法定位/读不到帧、
  无效 crop、写入失败/输出未生成、ffmpeg 失败、完整/部分断点续跑。
- 所有修改和新增 Python 文件语法检查通过，`WASD_test.py --help` 正常。

未执行完整 AVA/WASD 数据集评测，未重新测量 mAP；上述结论来自源码核对和合成输入回归，
不声称已重现完整数据集的评测分数。
