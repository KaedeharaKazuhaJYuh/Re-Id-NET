# Mamba–Transformer ReID · 原文重建与路线 C

周非、蒲淋、刘清林，《多尺度特征融合的 Mamba-Transformer 行人重识别网络》，《电子测量与仪器学报》，2026-10-21 网络首发。
[原文首发记录](https://link.cnki.net/urlid/11.2488.TN.20261020.1723.021)。重建时直接核读了用户保存的 13 页原文 PDF，而非二手摘要。未找到可核验的论文作者代码、训练配置或权重；本包是独立重建，不是官方源码或正式结果复现。

## 已验证：本地 CPU 冒烟

默认仅使用程序合成纹理，不需要图片、账号、GPU 或 GitHub；可用 Python 3.12。可从 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择对应环境的 wheel，在虚拟环境中安装即可。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -v
python run_smoke.py
```

已用 PyTorch 2.6 CPU 验证 8 项测试，覆盖输入 256×128 时的四阶段空间尺寸、端到端有限梯度、P×K 采样、跨相机评估，以及四个损失选项的训练。快速模型在 64×32 合成纹理上各跑两步。附带 `smoke-results.json` 包含七组：五组对应表 2 的模块开关，另两组为路线 C。后四组用相同的完整结构、种子和优化预算。合成检索和训练排名诊断仅验证数据流，不作算法效果解释。

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

## CUDA / 经许可的 Market-1501：未在本环境执行

需要本地可正常工作的 CUDA PyTorch 和 [NVlabs MambaVision](https://github.com/NVlabs/MambaVision) 及其 CUDA selective-scan 依赖。按上游说明安装并核对 GPU 和 wheel 兼容性；包不自带或执行上游代码，也不自动下载数据、模型或更改远程仓库。数据从[数据作者的项目页](https://zheng-lab-anu.github.io/Project/project_reid.html)与其使用条件核对，仅传入已经获准使用的本地目录。预训练权重是 [NVIDIA Tiny ImageNet-1K](https://huggingface.co/nvidia/MambaVision-T-1K)，不是周非等人的 ReID 权重。

先用原结构、一轮、不开官方测试集跑预算预检：
```sh
python train.py --backend official-tiny --device cuda --market-root /your/local/Market-1501-v15.09.15 --imagenet-checkpoint /your/local/mambavision_tiny_1k.pth.tar --epochs 1 --skip-eval --out preflight.json
```
若显存不足，优先同步降低所有对照的 P，不改变 K=4；记录配置和实测时间。随后在训练身份内独立选定未披露参数，不用官方测试集调参；当前包 **未实现该验证集切分**。锁定设置后训完整原文基线和骨干对照、在官方 query/gallery 运行一次，记录 mAP 和 Rank-1。CLI 使用 `--structure backbone --variant none` 启动骨干对照；完整组默认 `--structure full --variant ratr`。日后用 `--variant none|ratr|cov|cov_consistency` 在同样的完整结构和三个种子上跑路线 C；`cov_consistency` 是交叉协方差正则加身份概率 JS 一致性，不是原论文损失。Duke/MSMT 不在本实现的数据入口里。

如需作者级别复现，最值得补的不是新 GitHub 仓库，而是作者代码/最终权重、RATR 温度与隐藏实现、末层权重迁移说明。正式模型没有跑出论文指标以前，不把 CPU 合成测试当成复现成功。
