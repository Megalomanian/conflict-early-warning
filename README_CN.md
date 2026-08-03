# 社交媒体评论流冲突升级的无监督风险预测

[![论文](https://img.shields.io/badge/论文-PDF-blue)](./main.pdf)
[![English](https://img.shields.io/badge/README-English-blue)](./README.md)
[![幻灯片](https://img.shields.io/badge/幻灯片-HTML-orange)](./slides.html)

本仓库包含论文源码、数据处理流程、无泄露实验和复现产物。上游冲突指数是**无监督**的：无需冲突标签，综合攻击强度、负面高唤醒情绪和立场极化三个信号。下游则分别评估连续值预测和高风险状态分类，二者不可混为同一项结果。

## 主要结论

| 场景 | 任务 | 留出集结果 | 结论 |
|---|---|---:|---|
| 合成轨迹 | 连续值预测 | CNN-BiLSTM R² **0.809 ± 0.010** | 模型能学习受控动态 |
| 微博 27 个反转事件 | 评论量预测 | R² **0.715** | 在密集真实事件流上得到正向结果 |
| 知乎 20 个话题 | 原始冲突指数预测 | R² **≈ 0** | 稀疏文本轨迹的精确值不可可靠预测 |
| 知乎最终 25% | 清洗后高风险分类 | F1 **0.778**，召回率 **0.822** | F1/召回率优于 Persistence，但 AUC 未超过它 |

知乎结果的边界需要明确：Logistic 的 AUC 为 0.891，Persistence 为 0.896。因此正向结论是检测均衡性和召回率提高，而不是所有指标全面领先，也不是连续冲突值已被准确预测。

## 实验架构

```text
原始评论
  ├─ 中文 BERT 情感模型 → 攻击强度 + 负面高唤醒情绪
  └─ MiniLM 语义向量 → 仅训练集 K-means → 立场极化
                ↓
按话题、仅训练集拟合 ECDF 标定
                ↓
加权冲突分数（0.5 攻击 + 0.3 情绪 + 0.2 立场）
                ↓
规则时间箱内 Top-15% 聚合 → 冲突轨迹
                ↓
时序窗口（观察 12 箱 → 预测未来 6 箱）
  ├─ 连续值：统计、机器学习和神经网络预测器
  └─ 风险状态：因果状态历史特征 → Logistic 回归
```

CNN-BiLSTM 由两层一维卷积（32 通道，卷积核 3 和 5）、两层双向 LSTM（每方向 64 隐单元，dropout 0.2）和六步线性输出头组成。知乎使用 12 小时时间箱，因此 12 步输入对应过去 6 天，6 步输出对应未来 3 天。

## 无数据泄露评估协议

权威 V2 流程严格保持时间因果性：

1. 每个话题映射到规则时间网格，并按时间顺序划分为 60% 训练、15% 验证和 25% 最终测试。
2. K-means、每话题 `QuantileTransformer` ECDF、缩尾、缺失值填补、标准化和风险阈值都只由训练期观测拟合。
3. 验证集 F1 选择因果 EWMA 跨度、特征组、Logistic `C` 和决策阈值。最终选择为 24 箱（12 天）、仅状态历史特征、`C=0.1`、阈值 0.4。
4. 丢弃跨越划分边界的窗口；样本从不随机打乱；最终测试段不参与模型或超参数选择。
5. `preprocessing_audit.json` 记录每个话题的拟合边界；20 个评估话题的测试评论参与拟合数均为 0。

最终知乎分类包含 1,428 个训练窗口、680 个验证窗口和 1,706 个测试窗口。话题级 Bootstrap 得到 Logistic 相对 Persistence 的 ΔF1 **+0.062**（95% CI 0.018–0.108）、Δ召回率 **+0.241**（0.173–0.317）和 ΔAUC **−0.0046**（−0.0079 至 −0.0013）。20 个话题中，15 个 F1 提升，18 个召回率提升。

## 仓库结构

```text
main.tex, main_cn.tex              中英文论文源码
srep_submission/                   Scientific Reports 投稿版
run_experiments_v2.py              合成数据多种子基准（权威）
experiment_real_model_v2.py        文本信号与真实数据实验（权威）
optimize_real_signal_v2.py         知乎无泄露风险模型选择
case_study.py, eval_triggers.py     微博案例与预警触发规则
figures/                            论文图
experiment_results_v2/             指标、审计、日志和序列化结果
zhihu_topics/                       知乎与微博反转数据
reading_list/                       文献阅读笔记
```

不带 `_v2` 后缀的旧实验脚本采用会泄露未来信息的随机划分，已经废弃，不应引用其输出。

## 环境配置与复现

官方环境使用 Python 3.13+，由 [`uv`](https://docs.astral.sh/uv/) 管理。程序会自动检测 CUDA；可以回退到 CPU，但 Transformer 编码和神经网络基线会明显变慢。

```bash
uv sync
uv run python3 run_experiments_v2.py
uv run python3 experiment_real_model_v2.py
uv run python3 optimize_real_signal_v2.py
uv run python3 case_study.py
uv run python3 eval_triggers.py
```

必须先运行 `experiment_real_model_v2.py`，再运行 `optimize_real_signal_v2.py`，因为后者读取 `experiment_results_v2/trajectories_real_model.pkl`。Hugging Face 模型需要已缓存或可联网下载；设置 `HF_HUB_OFFLINE=1` 可强制只读取本地缓存。

关键复现产物包括：

- `aggregated_results.pkl`、`seed_results.pkl`：五种子合成实验汇总和逐次结果。
- `preprocessing_audit.json`：逐话题预处理无泄露审计。
- `signal_cleaning_results.json`：入选超参数、测试指标、敏感性分析和 Bootstrap 区间。
- `leakage_free_gpu_run_20260803.tar.gz`：无泄露 GPU 运行归档，包含日志与数据产物。

## 基线与评价指标

合成实验在 5 个随机种子上比较 Persistence、Moving Average、Exponential Smoothing、AR(6)、SVR、XGBoost、TCN、Informer-Lite、BiLSTM、Transformer、BiGRU 和 CNN-BiLSTM，报告 R²、MAE、RMSE、升级事件 F1 和基于事件的提前量。真实连续值实验同时报告 Persistence 参照；分类实验报告 AUC、精确率、召回率和 F1。

在主设定的训练期第 80 百分位风险定义下，Logistic 的精确率、召回率和 F1 分别为 0.738、0.822 和 0.778；Persistence 分别为 0.926、0.584 和 0.716。第 70、80、90 百分位敏感性分析中的 Logistic F1 分别为 0.779、0.778 和 0.761。

## 编译论文

```bash
./compile.sh en       # main.tex → main.pdf
./compile.sh cn       # main_cn.tex → main_cn.pdf
./compile.sh clean    # 清理辅助文件
```

编译脚本调用 `~/.local/bin/tectonic` 并自动处理 BibTeX。提交论文改动前，应重新生成对应 PDF，并检查引用、图片和表格是否正常解析。

## 引用

```bibtex
@article{zhu2025conflict,
  title={Unsupervised Risk Forecasting of Conflict Escalation in Social Media Comment Streams},
  author={Zhu, Linli and Ma, Ziqiang},
  journal={IEEE Transactions on Computational Social Systems},
  year={2025},
  note={Under review}
}
```
