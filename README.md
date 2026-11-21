# Mamba–Transformer ReID · 原文重建与路线 C

周非、蒲淋、刘清林，《多尺度特征融合的 Mamba-Transformer 行人重识别网络》，《电子测量与仪器学报》，2026-10-21 网络首发。
[原文首发记录](https://link.cnki.net/urlid/11.2488.TN.20261020.1723.021)。重建时直接核读了用户保存的 13 页原文 PDF，而非二手摘要。未找到可核验的论文作者代码、训练配置或权重；本包是独立重建，不是官方源码或正式结果复现。

## 当前复现状态

**三种子初步结果：** 固定 30 步、50 个留出查询，三次配对中 RATR 的拼接 mAP 均低于无 RATR。平均 mAP / Rank-1：无 RATR **53.73% / 69.33%**，有 RATR **50.72% / 63.33%**，差值 **−3.01 / −6.00 个百分点**。这是同一内部验证拆分的短预算结果，不是官方测试成绩。

**Market-1501 已在 RTX 5070 Ti Laptop GPU 上通过断点续训及 RATR 分项短测；尚未完成整轮训练，也未在官方测试集评估。** 这些试跑只验证训练、断点和损失实现能工作，不用于和论文精度对比。

| 验证项 | 结果 |
|---|---|
| Mamba CUDA 算子 | 合成输入的前向输出与六组反向梯度均通过 CPU 对照，最大绝对误差约 `7.45e-9` |
| 完整模型 | 合成小批次完成前向、反向和一次优化器更新 |
| Market-1501 链路短测 | NVIDIA Tiny ImageNet-1K 初始化；完整结构，P2×K2；第 1–10 步平均损失 19.0618，第 11–20 步 18.4360。选了 RATR，但排序候选不足，两项排序损失均为零；PyTorch 峰值分配约 1.08 GiB |
| Market-1501 RATR 短测 | 同一预训练初始化，另起 P3×K4 任务跑 10 步；平均总损失 21.3281，其中类内 RATR 0.2985、类间 RATR 0.1849；训练段 6.81 秒、PyTorch 峰值分配约 2.16 GiB；另存断点 |
| 断点续训 | 笔记本已验证从第 10 步恢复并更新到第 20 步本地断点；CPU 合成测试验证轮中、轮末恢复均与连续运行一致。CUDA 续跑与不停跑是否逐位一致尚未验证 |

短测均未读取 query/gallery。P2×K2 中，每张图只有一个同身份候选和一个负类质心，不能构造 RATR 排序对。P3×K4 验证两项都有贡献，但与 P2×K2 的损失不能直接对比，也不替代论文的 P16×K4 对照。数据、权重和断点均未上传。

## 预注册探索性对照：30 步训练内验证

采用同一完整模型、同一 ImageNet Tiny 初始化、AdamW（基础学习率 3.5e-4，权重衰减 0.05）、训练种子 7、P16×K4，仅比较 `--variant none` 与 `--variant ratr`。两组各训练至累计第 30 步，`--epochs 160 --warmup 1`，首轮学习率为 3.5e-4。这里将 warmup 从论文的 20 个 epoch 缩至 1 个 epoch，只为了让固定小预算试验具有诊断意义，不是论文训练方案。各组使用独立的 checkpoint 和结果路径；每 10 步输出训练摘要。

训练及验证只读 Market-1501 的 `bounding_box_train`。以拆分种子 20261121 选出 50 个至少由两个摄像头拍摄的身份；每个身份选一张作查询，其余作为验证库，所有留出身份不参与训练。固定拆分与指标；在第 30 步保存 checkpoint 后，每组另启动一次进程进行验证，不再执行任何优化器步。官方 query/gallery 不读取。仅报告两组训练摘要和训练内验证 mAP / Rank-1；50 个查询、单种子、30 步不足以判断方法优劣，也不用于复现论文分数。

各组流程（`VARIANT` 分别设为 `none` 和 `ratr`，`RUN_DIR` 必须是彼此不同且已存在的路径）：

```sh
COMMON=(--backend official-tiny --device cuda --market-root "$MARKET" \
  --imagenet-checkpoint "$IMAGENET" --structure full --variant "$VARIANT" \
  --p 16 --k 4 --seed 7 --epochs 160 --warmup 1 --max-steps 30 \
  --validation-identities 50 --validation-split-seed 20261121 \
  --save-every-steps 10 --progress-every 10)
python train.py "${COMMON[@]}" --resume-state "$RUN_DIR/state.pt" --skip-eval \
  --out "$RUN_DIR/train-summary.json"
python train.py "${COMMON[@]}" --resume-from "$RUN_DIR/state.pt" \
  --eval-validation-at-limit --out "$RUN_DIR/holdout-step30.json"
```

先前按论文形状运行的容量预检已完成 P16×K4 的第 10 步，峰值 CUDA 已分配显存约 9.27 GiB。该预检未运行检索评价，不是上述两组对照的结果。

### 固定 30 步初步结果（内部验证）

两组拆分指纹完全一致：701 个训练身份、12,142 张训练图；50 张查询图、744 张验证库图，50 个有效查询。评估均从第 30 步断点加载，不再更新参数。mAP 使用保留排序中命中位置的精度均值，仅对存在有效匹配的查询取平均；不做重排序。论文未公开足以核对的 AP 计算细节。

| 检索特征 | 无 RATR mAP / Rank-1 | 有 RATR mAP / Rank-1 |
|---|---:|---:|
| 骨干 | 52.27% / 64% | 53.12% / 66% |
| 融合分支 | 54.77% / 70% | 55.10% / 72% |
| 拼接（主要指标） | 55.48% / 70% | 54.83% / 70% |

拼接 mAP 下降约 0.65 个百分点，Rank-1 不变；单支点估计略高，但没有转化成拼接收益。本试验单种子、50 查询、30 步，不能归因为 RATR 普遍有效或无效，也不能与论文表格直接比较。未触碰官方 query/gallery；权重、数据、断点与逐样本结果未上传。


### 两个额外种子复测

在不改变第一个种子的留出拆分、优化预算、初始化来源、批次或指标的前提下，补跑训练种子 17 和 29。所有运行的验证拆分指纹一致；各组仍只训练 30 步。

| 训练种子 | 无 RATR mAP / Rank-1 | 有 RATR mAP / Rank-1 | 拼接 mAP 差值（百分点） |
|---|---:|---:|---:|
| 7 | 55.48% / 70% | 54.83% / 70% | −0.65 |
| 17 | 53.80% / 72% | 48.26% / 58% | −5.55 |
| 29 | 51.90% / 66% | 49.07% / 62% | −2.83 |
| 三种子平均 | **53.73% / 69.33%** | **50.72% / 63.33%** | **−3.01** |

这支持一个有限结论：在固定 30 步和这一留出拆分上，未看到 RATR 提升最终拼接检索。三个训练种子并不是三个独立验证集，不能外推到论文的完整训练、官方测试成绩或 RATR 的一般有效性。没有为追求正向结果更改配置。

## CPU 快速验证

默认仅使用程序合成纹理，不需要图片、账号、GPU 或 GitHub；可用 Python 3.12。可从 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择对应环境的 wheel，在虚拟环境中安装即可。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -v
python run_smoke.py
```

已用 PyTorch 2.6 CPU 通过 12 项测试，覆盖四阶段空间尺寸、端到端有限梯度、P×K 采样、跨相机评估、四个损失选项、断点续训和保存失败保护。快速模型在 64×32 合成纹理上各跑两步。附带 `smoke-results.json` 包含七组：五组对应表 2 的模块开关，另两组为路线 C。后四组用相同的完整结构、种子和优化预算。合成检索和训练排名诊断仅验证数据流，不作算法效果解释。

## 原文对照

| 部分 | 原文已核实 | 本包仍需核对的工程选择 |
|---|---|---|
| 骨干 §1.1 | Tiny 预训练；第三、四阶段间用 1×1、320→512；深层 16×8 | [NVlabs Tiny](https://github.com/NVlabs/MambaVision) 默认末层为 640。CUDA 适配器加载匹配的 stem 与前三阶段，将 512 通道第 4 阶段重新初始化。作者未说明末层预训练权重如何迁移，未猜测其策略。 |
| HCEN §1.2、图 1/3、表 4/5 | 三个级联节点，Mixer 深度 1/2/3，分流回注与浅层去噪 | 各层融合宽度、具体对齐算子、残差后合并未给可复现配置；本包以第一节点去噪、平均池化对齐、相加合并，CUDA 默认宽度 512。 |
| SASF §1.2 | 3×3/5×5 膨胀深度可分离卷积、学习权重和输入残差 | 空洞率 2、权重 softmax 是本包选择。 |
| AUL §1.3 式 15–18 | label smoothing .1，身份＋三元组，辅助权重 .3 | 只给 G2/G3 辅助头，不重复计最后的 G4；三元组用 Euclidean batch-hard，margin .3 为本包选择。 |
| RATR §1.3 式 19–24 | 两个 tanh 的乘积取有符号均值；排除自身的正例排序与负类归一化质心排序；两个均值直接相加；默认权重 1 | 温度未报告，.1 是本包占位。度量/RATR 取 BN 前特征，检索取 BN 后特征；最终用分支分别归一化后拼接，是未获作者确认的选择。 |
| 训练 §2.2 | 256×128、P16×K4、AdamW、lr 3.5e-4、wd .05、warmup 20、cosine、160 epochs；随机翻转、裁剪、擦除 | 增强参数、warmup 具体起始倍率、精确采样器和评估代码未获核对。 |

原始 NumPy 原型不是 RATR 公式实现：它没有对负类质心再次做 L2 归一化，并把正负两项各乘了 0.5。本包按原文两个分项**直接相加**，说明温度与分支特征取法仍是未确认项。若两支排名方向正好相反，单独最小化有符号相关的确会给较低值；这不等于联合损失训练必然破坏身份判别。

表 2 的 Market-1501（mAP/Rank-1）是骨干 88.3/94.9、完整 89.9/95.6。这是论文参考值，不是本包的运行结果。§2.3 的表后正文对 MSMT17/Duke 单独 HCEN 的增量与表 2 交换了顺序；以表列的原始分数为准（MSMT17 63.1→64.4；Duke 81.9→82.7）。

## CUDA 与 Market-1501 短程验证

Windows 上已验证的环境是 Docker Desktop 的 WSL 2 Linux 容器：Python 3.11、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、[NVlabs MambaVision](https://github.com/NVlabs/MambaVision) 1.2.0，以及 [Mamba CUDA 预编译包](https://github.com/state-spaces/mamba/releases/tag/v2.2.6.post3) 2.2.6.post3（Linux x86-64、CPython 3.11、CXX11 ABI true）。这里有意替换 NVlabs 声明的 `mamba-ssm==2.2.4`，不要让通用安装命令将其覆盖。该 PyTorch 容器内另补了 GCC 和 libc 开发头文件供 Triton 初始化小型 C 辅助模块；没有从源码编译 Mamba CUDA 扩展。

数据请从[发布者项目页](https://zheng-lab-anu.github.io/Project/project_reid.html)核对使用条件，并使用本地数据目录。预训练权重来自 [NVIDIA Tiny ImageNet-1K](https://huggingface.co/nvidia/MambaVision-T-1K/blob/main/model.safetensors)，不是周非等人的 ReID 权重。已验证的做法是读取 NVIDIA 的 `model.safetensors`（SHA-256 `4f6987be0a2ca2222f386eb750d028a05203b047d3c8dfb664c27e2295d02fc0`），去掉键名的 `model.` 前缀，再将纯张量字典存为本地 `.pt`。训练脚本仍使用 `weights_only=True` 读取；原版 `.pth.tar` 包含额外训练对象，不能直接通过此安全加载方式。

先复跑 10 步预算预检，不开官方测试集：

```sh
mkdir -p runs
python train.py --backend official-tiny --device cuda \
  --market-root /your/local/Market-1501-v15.09.15 \
  --imagenet-checkpoint /your/local/mambavision_tiny_1k.weights-only.pt \
  --p 2 --k 2 --max-steps 10 --skip-eval \
  --resume-state runs/pilot.pt --save-every-steps 10 --progress-every 10 \
  --out runs/pilot.json
```

正式基线需新建任务、使用论文批次和计划，不把 P2×K2 的短测断点接到 P16×K4 上。若显存不足，优先同步降低所有对照的 P，不改变 K=4；记录配置和实测时间。随后在训练身份内独立选定未披露参数，不用官方测试集调参；当前包 **未实现该验证集切分**。锁定设置后训完整原文基线和骨干对照、在官方 query/gallery 运行一次，记录 mAP 和 Rank-1。CLI 使用 `--structure backbone --variant none` 启动骨干对照；完整组默认 `--structure full --variant ratr`。日后用 `--variant none|ratr|cov|cov_consistency` 在同样的完整结构和三个种子上跑路线 C；`cov_consistency` 是交叉协方差正则加身份概率 JS 一致性，不是原论文损失。Duke/MSMT 不在本实现的数据入口里。

如需作者级别复现，最值得补的不是新 GitHub 仓库，而是作者代码/最终权重、RATR 温度与隐藏实现、末层权重迁移说明。没有正式评估前，不把 CPU 验证或小批量短测当成论文指标复现。

## 早期诊断

`diagnostics/numpy-route-c/` 保留第一版 NumPy 合成诊断与结果，仅供比较；正式实现以仓库根目录的 PyTorch 版本为准。

## 断点续训与进度

正式启动前保持相同的数据和训练参数（尤其是总轮数、warmup、P/K 和 seed）。下例可用 CPU 验证操作，不读真实图片：

```sh
mkdir -p runs
python train.py --epochs 2 --p 2 --k 2 --max-steps 10 --skip-eval \
  --resume-state runs/resume.pt --save-every-steps 10 --progress-every 10
python train.py --epochs 2 --p 2 --k 2 --skip-eval \
  --resume-from runs/resume.pt --save-every-steps 10 --progress-every 10
```

- 新任务用 `--resume-state`，恢复用 `--resume-from`；默认写回原断点。每 10 步、每轮结束和达到显式步数上限时保存模型、优化器、随机状态和数据游标。强制停止后从最近写完整的断点恢复，之后未保存的步数会重跑。只加载自己任务产生的本地断点。
- `--max-steps` 是整个任务的累计上限，不是本次额外运行的步数。继续到第 20 步须把上限改为 20，或省略它继续至既定轮数。恢复时不能改训练配置或在两个进程同时写同一断点。
- `--progress-every 10` 实时输出一行 JSON（累计步数、区间平均损失、学习率、CUDA 峰值显存）。`training_seconds` 和峰值显存只代表当前进程段。断点与结果留在本地，勿提交数据或权重。
- CPU 合成验证比较了中途停止和轮末停止两种情况；恢复后的模型、优化器状态和损失日志与连续运行一致。CUDA 上的逐位重现尚未验证。
