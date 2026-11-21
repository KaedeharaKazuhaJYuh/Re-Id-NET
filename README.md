# Mamba–Transformer ReID · 原文重建与路线 C

周非、蒲淋、刘清林，《多尺度特征融合的 Mamba-Transformer 行人重识别网络》，《电子测量与仪器学报》，2026-10-21 网络首发。
[原文首发记录](https://link.cnki.net/urlid/11.2488.TN.20261020.1723.021)。重建时直接核读了用户保存的 13 页原文 PDF，而非二手摘要。未找到可核验的论文作者代码、训练配置或权重；本包是独立重建，不是官方源码或正式结果复现。

## 当前结论

- Market-1501 已在 RTX 5070 Ti 上跑通训练和断点续训，尚未完成整轮训练，也没有官方测试集成绩。
- 路线 C 在第一轮短训有收益，但换一组不重叠的留出身份后未复现。暂不把它视为稳定提升。
- 逐种子结果、数据拆分、复现参数和运行记录见 [result.md](result.md)。

## 快速验证

在已安装 PyTorch 的 Python 环境中运行。以下命令默认用合成数据，不需要 GPU 或 Market-1501。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -v
python run_smoke.py
```

真实数据运行请先看 [CUDA 环境与短跑说明](result.md#cuda-与-market-1501-短程验证) 和 [断点续训](result.md#断点续训与进度)。
