# 社交媒体评论流冲突升级的无监督风险预测

[![论文](https://img.shields.io/badge/论文-PDF-blue)](./main.pdf)
[![幻灯片](https://img.shields.io/badge/幻灯片-HTML-orange)](./slides.html)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

**作者**：朱林立 (Linli Zhu)<sup>1</sup>，马自强 (Ziqiang Ma)<sup>2</sup>  
<sup>1</sup> 宁夏大学计算机学院  
<sup>2</sup> 宁夏大学信息工程学院  
📧 violina2333@gmail.com · maziqiang@nxu.edu.cn

## 概述

一种面向社交媒体评论流冲突升级的**无监督**风险预测框架。该框架从**攻击强度**、**负面高唤醒情绪**和**立场极化**三个可解释信号出发，在无需人工标注的条件下构建评论级冲突指数，进而使用深度序列预测器进行短时域时序预测。论文在严格的时序划分协议下，通过三个互补场景对框架进行评估，诚实报告正面和负面结果。

### 核心发现（V2 — 严格时序划分）

| 场景 | 预测目标 | R² | 含义 |
|------|---------|-----|------|
| 合成冲突轨迹 | 冲突指数 | **0.809 ± 0.010** | 理想条件下架构有效 |
| 微博反转事件（27个池化） | 评论量 | **0.715** | 预测器在真实密集事件上可泛化 |
| 知乎讨论（20个话题） | 冲突指数 | **≈ 0** | 12h粒度的文本冲突信号不可预测 |

**核心观点**：冲突指数揭示了一个事实——粗粒度下的文本冲突信号缺乏短期可预测性。这个负面结果是信息性的——它定义了随机划分协议系统性地掩盖的边界条件。

### 架构

```
原始评论 → 冲突指数 → 时间箱轨迹 → 深度预测器 → 风险监测
        （3信号）   （top-k聚合）  （L=12, H=6）
```

## 仓库结构

```
├── main.tex / main_cn.tex    # 中英文论文（IEEEtran）
├── srep_submission/           # Scientific Reports 投稿版
├── refs.bib                  # 参考文献（29条）
├── slides.html               # reveal.js 演示文稿
├── compile.sh                # 编译：./compile.sh [en|cn|clean]
├── AGENTS.md                 # 贡献者指南
│
├── run_experiments_v2.py           # 主实验：合成数据 + 全基线（时序划分，5种子）
├── experiment_real_model_v2.py     # 真实数据：BERT冲突指数 + CNN-BiLSTM
├── case_study.py                   # 微博反转事件案例研究（V2已修复）
├── eval_triggers.py                # 触发规则评估
│
├── reproduce_competitors/          # 演示：随机划分 vs 时序划分的性能膨胀
│
├── run_experiments.py / experiment_real_model.py  # 已废弃（随机划分数据泄露）
│
├── figures/                  # 论文图表（8张PDF）
├── experiment_results_v2/    # V2实验输出（.pkl文件）
├── reading_list/             # 精选文献及摘要
├── zhihu_topics/             # 知乎数据集（140话题）+ weibo_reversal/子数据集
│
├── pyproject.toml            # Python 依赖（uv 管理）
└── .vscode/settings.json     # LaTeX Workshop 自动编译配置
```

**⚠️ `run_experiments_v2.py` 和 `experiment_real_model_v2.py` 是权威脚本。** 不带 `_v2` 的旧脚本使用随机划分（存在数据泄露），已废弃。

## 构建与运行

### 论文编译

```bash
./compile.sh en          # 英文 → main.pdf（Tectonic，自动 BibTeX）
./compile.sh cn          # 中文 → main_cn.pdf
./compile.sh clean       # 清理辅助文件
```

需要 **Tectonic**，路径为 `~/.local/bin/tectonic`。

### 实验运行

```bash
uv sync                                          # 安装依赖
uv run python3 run_experiments_v2.py             # 主实验（推荐GPU，约90分钟，5种子）
uv run python3 experiment_real_model_v2.py       # 真实数据冲突指数（需GPU）
uv run python3 eval_triggers.py                  # 触发规则评估
uv run python3 reproduce_competitors/demo_random_vs_temporal.py  # 数据泄露演示
```

**环境要求**：Python 3.13+，PyTorch 2.13+，Transformers，Sentence-Transformers，XGBoost，scikit-learn。

### 演示文稿

在浏览器中打开 `slides.html`——基于 [reveal.js](https://revealjs.com/)。按 `F` 全屏，`?` 查看快捷键。

## 数据集

| 数据集 | 描述 | 规模 |
|--------|------|------|
| **合成数据** | Logistic升级事件 + 季节性 + 粉红噪声 | 60条轨迹 × 200 bins |
| **知乎** | 中文问答平台，评论稀疏 | 140个话题，4.5万条评论 |
| **微博反转** | 舆论反转事件，评论密集 | 27个事件，24.5万条评论 |

## 基线模型与结果（V2 — 合成数据）

| 模型 | 类型 | R²（均值±标准差） | Esc-F1 |
|------|------|-------------------|--------|
| Persistence | 统计 | 0.638 ± 0.015 | 0.637 |
| Moving Avg (k=3) | 统计 | 0.540 ± 0.020 | 0.547 |
| Exp. Smoothing | 统计 | 0.498 ± 0.022 | 0.430 |
| AR(6) | 统计 | 0.715 ± 0.011 | 0.644 |
| SVR (RBF) | 机器学习 | 0.759 ± 0.024 | 0.667 |
| XGBoost | 机器学习 | 0.781 ± 0.013 | 0.684 |
| TCN | 深度学习 | 0.787 ± 0.010 | 0.695 |
| Informer-Lite | 深度学习 | 0.796 ± 0.014 | 0.659 |
| BiLSTM | 深度学习 | 0.805 ± 0.011 | 0.711 |
| Transformer + PE | 深度学习 | 0.806 ± 0.007 | 0.697 |
| **BiGRU** | 深度学习 | **0.811 ± 0.009** | **0.716** |
| **CNN-BiLSTM（本文）** | 深度学习 | **0.809 ± 0.010** | **0.710** |

严格时序划分（无打乱），ECDF仅在训练数据上拟合，5个独立随机种子。BiGRU与CNN-BiLSTM在统计上不可区分（ΔR² = -0.002，在1σ以内）。

## 真实数据结果（V2 — 时序划分）

| 数据集 | 预测目标 | R² |
|--------|---------|-----|
| 微博27事件，1h bins | 评论量（池化） | **0.715** |
| 知乎20话题，12h bins | 冲突指数 | ≈ 0 |
| 知乎（Persistence基线） | 冲突指数 | -0.90 |
| 知乎（AR(6)基线） | 冲突指数 | ≈ 0 |

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

## 许可证

MIT License — 详见 [LICENSE](./LICENSE) 文件。
