# 论文阅读清单

为论文 "Weakly Supervised Early Warning of Conflict Escalation in Social Media Comment Streams" 推荐的阅读文献，按主题分类。

---

## 1. 直接相关：社交媒体冲突预测 + LSTM

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 1.1 | **CUP_CDLSTM: Civil Unrest Event Prediction Using CNN, DistilBERT, and LSTM** | *IEEE Trans. Computational Social Systems*, 2025 | CNN-LSTM + DistilBERT 预测香港抗议和 BLM 事件；跨地域泛化，比基线高 5% |
| 1.2 | **SatCoBiLSTM: Self-attention based hybrid deep learning framework for crisis event detection in social media** | *Expert Systems with Applications*, 2024 | Self-Attention + Bi-LSTM 混合框架做危机事件检测，有 2025 年后续扩展 |
| 1.3 | **Social LSTMs for Inter-Community Conflict Prediction** | Stanford 研究项目 | 社区嵌入 + LSTM 预测 Reddit 社区间冲突，开源 PyTorch 实现 |
| 1.4 | **Affective Homophily as the Dominant Organizing Principle in Online Conflict Discourse Networks** | *Scientific Reports*, 2025 | GATv2 链接预测分析微博冲突话语网络，发现情感同质性优于结构声望 |

---

## 2. 弱监督/无监督冲突信号构建

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 2.1 | **Ex Machina: Personal Attacks Seen at Scale (Wulczyn et al.)** | *WWW 2017* | 已引用。构建 Wikipedia 攻击标注数据，输出连续攻击概率——冲突指数的核心参考 |
| 2.2 | **SafeSpeech: a three-module pipeline for hate intensity mitigation** | *Social Network Analysis and Mining*, 2024 | 超越二分类，引入仇恨强度连续评分（hate intensity identification），印度语言 |
| 2.3 | **When comments aren't what they seem: ICL-DiTox** | *Expert Systems with Applications*, 2025 | GPT-4o/DeepSeek 做多轮对话毒性检测，处理反讽和谐音，多语言 |
| 2.4 | **HateModerate: Testing Hate Speech Detectors against Content Moderation Policies** | *NAACL 2024* | 检测器与真实平台政策对齐，强调策略合规性 |

---

## 3. 立场检测与极化度量（无监督）

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 3.1 | **Transformer-Based Quantification of the Echo Chamber Effect in Online Communities** | UC3M, 2024 | 用 Transformer 嵌入量化回音室效应和极化程度 |
| 3.2 | **A Framework for the Unsupervised Modeling and Extraction of Polarization Knowledge from News Media** | *Semantic Scholar* | 无监督方法从新闻媒体提取极化知识 |
| 3.3 | **STEntConv: Predicting Disagreement with Stance Detection and a Signed Graph Convolutional Network** | *arXiv*, 2024 | 符号图卷积网络 + 立场检测预测分歧 |
| 3.4 | **Political Leaning Inference through Plurinational Scenarios** | *arXiv*, 2024 | 多国家场景下的政治倾向推断，嵌入聚类方法 |

---

## 4. LSTM 舆论时序预测（2024-2025 最新）

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 4.1 | **TSTL: A Hybrid TextRank-LP and TDNN-LSTM Model for Public Opinion Monitoring** | *Informatica*, 2025 | TextRank + TDNN-LSTM 预测舆情阶段，关键词提取 95.1%，预测时间 21 分钟 |
| 4.2 | **BERT-TextCNN + Att-LSTM for Public Sentiment Prediction** | *Modern Electronics Technique*, 2025 | BERT-TextCNN 做情感分类 + Att-LSTM 时序预测，RMSE 0.084 |
| 4.3 | **Modeling Group-Level Public Sentiment through Topic and Role Enhancement (TRESP)** | *Knowledge-Based Systems*, 2024 | 话题+角色增强的群体情感预测，层次注意力网络 |
| 4.4 | **The Memory Cycle of Time-Series Public Opinion Data** | *Information Processing & Management*, 2025 | 发现舆论时序的多尺度周期性（短/中/长期），验证 CNN-LSTM 预测 |
| 4.5 | **Constructing Public Opinion Crisis Prediction Model Using CNN and LSTM** | *IJIMAI*, 2024 | CNN-LSTM 舆情危机预测，准确率 92.19% |
| 4.6 | **IPSO-LSTM Hybrid Model for Predicting Online Public Opinion Trends (Mu et al.)** | *PLOS ONE*, 2023 | 已引用。粒子群优化 LSTM 超参数，改进舆情预测 |

---

## 5. 信息级联与传播预测（GNN 路线）

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 5.1 | **ALETHEIA: Combating Social Media Influence Campaigns with GNNs** | *arXiv*, 2025 | GraphSAGE + RNN 做时序链接预测检测水军，AUC 96.6% |
| 5.2 | **D²: Two-Stage GNN for Early Rumor Detection through Cascade Diffusion Prediction** | *ACM*, 2025 | 动态异质 GNN 预测传播路径，可微分架构搜索 |
| 5.3 | **HyperIDP: Hypergraph-Based Information Diffusion Prediction** | *ACL*, 2025 | 超图神经网络做信息传播预测，合作-对抗损失处理多任务 |
| 5.4 | **CTPDN: Community Aware Temporal Pattern Diffusion Network** | *PatternIQ Mining*, 2025 | 社区感知时序扩散网络，早期级联检测 20-27% 提升 |

---

## 6. 预警触发规则与异常检测

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 6.1 | **EMBERS: Real-Time Civil Unrest Forecasting System** | Virginia Tech, 2025 | 大规模实时社会动荡预测系统，动态查询扩展，关注前置时间和精确率 |
| 6.2 | **Identifying Critical Outbreak Time Window of Controversial Events (Wang et al.)** | *PLOS ONE*, 2020 | 已引用。基于情感分析识别争议事件关键爆发时间窗 |

---

## 7. 综述与趋势

| # | 论文 | 出处 | 要点 |
|---|------|------|------|
| 7.1 | **Deep Learning for Sentiment Analysis: A Survey (Pimpalkar et al.)** | *IJACTE*, 2025 | 已引用。从经典方法到神经架构的情感分析综述 |
| 7.2 | **AI-Driven Social Media Text Analysis During Crisis** | *Applied Soft Computing*, 2025 | 危机情境下社交媒体文本分析的 AI 方法全面综述 |

---

## 关键趋势总结

1. **混合架构**：LSTM + Transformer（BERT/DistilBERT）+ Attention 成为主流
2. **连续风险评分**：从二分类转向细粒度连续评分（0-1），与本文冲突指数方向一致
3. **无监督/弱监督**：减少人工标注依赖，利用预训练模型做伪标签
4. **图神经网络融合**：信息级联预测中 GNN 成为标配，考虑引入用户关系图
5. **实时部署**：从离线实验转向流式在线推理和实际系统部署

---

*检索日期：2025-05-23*
