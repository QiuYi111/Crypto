加密货币日间交易强化学习智能体开发说明 (CryptoRL Agent Development Spec)
1. 项目概述
本项目的目标是开发一个先进的强化学习 (RL) 智能体，用于在Binance平台进行加密货币合约的日间交易 (T+1)。该智能体将融合大型语言模型 (LLM) 对新闻和情绪的理解能力，以生成多维度的“信心向量”，从而增强RL的观测空间。通过在多个加密货币标的（如BTC, ETH, SOL等）上进行多市场训练，旨在构建一个通用且鲁棒的交易策略，能够适应不同市场特性，实现收益最大化。特别地，我们将探索使用Mamba模型架构作为RL智能体的基座，并尝试多种主流RL训练算法。

2. 项目目标
实现日间交易收益最大化： 在加密货币合约市场中，通过RL策略实现稳健且可观的日间交易收益。

提升策略鲁棒性与泛化能力： 训练RL智能体在多个加密货币标的（而非单一市场）上学习通用交易逻辑，使其能够适应市场变化并可能泛化到新的或未训练过的资产。

融合非结构化信息： 利用LLM将海量历史新闻和事件转化为结构化的“信心向量”，为RL提供超越传统技术指标的深度市场洞察。

探索前沿模型架构： 尝试将Mamba模型架构应用于RL智能体，以期在处理时间序列数据和长序列依赖方面获得优势。

构建可扩展的AI交易系统： 搭建模块化、可维护的系统架构，便于后续的功能扩展、算法升级及性能优化。

3. 核心功能模块
本项目将分为以下几个主要功能模块：

3.1. 数据管道与预处理模块
功能： 负责所有历史和实时市场交易数据的获取、存储和初步处理。

数据源：

市场数据： 通过Binance API获取所有目标加密货币对的历史日线、小时线或4小时线K线数据 (OHLCV)，以及账户信息、合约资金费率等。

新闻与事件数据： 不进行本地大规模实时抓取。而是通过集成外部搜索API（如Google Search API, SerpApi, CoinDesk API等），供LLM按需查询特定日期和标的的历史新闻。

存储： 建议使用高性能数据库（如InfluxDB/ClickHouse 用于时间序列数据, PostgreSQL 用于账户/订单数据）存储数据。

输出： 结构化的市场K线数据及相关技术指标。

3.2. 信息增强模块 (LLM + 搜索API)
功能： 利用LLM对新闻信息进行理解和特征提取，生成“信心向量”。

LLM选型与部署：

推荐： 本地部署或优化推理速度的云服务（例如，基于Llama系列、Mixtral、Gemma等开源模型进行微调），以降低推理延迟和成本。

微调： 使用加密货币和金融领域的历史新闻及事件数据对LLM进行微调，以提升其对金融上下文的理解和特定“信心维度”的提取能力。

检索增强生成 (RAG) 机制：

离线训练数据生成： 在RL训练前，针对所有历史日期和所有目标标的，批量执行RAG流程：

根据日期和标的，向外部搜索API查询相关历史新闻。

将检索到的新闻摘要或文本作为上下文输入给LLM。

LLM输出一个多维度的“信心向量”（例如，[基本面情绪、市场情绪、监管影响、技术创新影响、地缘政治风险]，各0-1评分）。

将生成的信心向量与对应的日期和标的ID存储起来，作为RL的训练数据。

在线预测： 在实盘运行时，每天（或RL智能体需要更新其认知时）执行RAG流程，获取最新（通常是前一天）的信心向量。

输出： 结构化的、与特定日期和标的关联的“信心向量”数据。

3.3. 数据融合模块
功能： 将市场K线数据与LLM生成的信心向量整合，形成RL模型可用的增强型观测空间数据。

实现：

根据日期和标的ID，将每日K线数据（开盘价、收盘价、成交量、技术指标等）与LLM生成的信心向量进行横向拼接。

对所有数值特征进行标准化或归一化处理。

为每个数据点添加当前交易标的的ID编码（如One-Hot Encoding或Embedding），以便RL模型区分不同资产。

输出： 包含了市场状态、技术指标、LLM信心向量和资产ID的增强型数据集，作为RL训练和实盘观测的输入。

3.4. 强化学习策略模块 (CryptoRL Agent)
功能： 训练一个能够泛化到多市场、多资产的RL智能体，并根据增强数据做出交易决策。

RL框架： 推荐使用 Stable Baselines3 (PyTorch) 或 Ray RLlib。

环境构建：

观测空间 (Observation Space): 包含历史K线数据、技术指标、LLM信心向量、账户状态（当前仓位、可用保证金、资金费率、历史收益/亏损）以及当前交易标的ID。

动作空间 (Action Space):

离散动作： 买入、卖出、持有、平仓，并可细化为不同杠杆倍数或仓位大小。

连续动作 (可选)： 更精细地调整仓位大小。

奖励函数 (Reward Function): 基于实际交易收益/亏损，并集成交易手续费、资金费率、滑点惩罚、风险管理（如对超额杠杆、最大回撤的惩罚）。

状态重置： 定义每个训练回合的结束条件（如达到最大交易天数、触发最大亏损、账户爆仓）。

RL算法探索：

基座架构： 尝试将Mamba架构作为Actor和Critic网络的骨干，以期在处理长序列依赖方面获得优势。

主流算法： 探索并对比多种主流RL算法的性能，包括但不限于PPO, SAC, TD3。

训练策略：

在增强数据集上进行训练，确保训练数据覆盖多市场、多时间段。

RL环境在训练回合中随机采样不同的标的历史数据进行学习，促进模型的泛化能力。

3.5. 交易执行与风险管理模块
功能： 将RL智能体的决策转化为实际交易指令，并严格执行风险控制。

交易API集成： 使用python-binance或其他稳定可靠的Binance API SDK。

订单类型： 主要使用限价单以控制滑点，并辅以止损 (Stop-Loss) 和止盈 (Take-Profit) 订单进行风险管理。

执行频率： 根据RL智能体每日的决策进行交易，执行频率为日级别（T+1）。

风险控制：

硬性限制： 严格执行预设的最大仓位限制、单笔交易最大亏损、账户最大回撤等。

资金管理： 确定每次交易的投入资金比例。

错误处理： 健壮的API调用错误处理、重试机制、网络中断处理。

3.6. 回测与监控模块
功能： 对策略进行离线性能评估，并对实盘交易进行实时监控。

离线回测：

使用历史增强数据集（包含LLM信心向量）进行高精度模拟回测。

评估指标： 夏普比率、索提诺比率、最大回撤、年化收益率、胜率、盈亏比等。

实时监控：

可视化仪表盘： 使用Streamlit/Dash等工具，实时显示账户资金、当前仓位、PNL、RL智能体的最新决策、LLM最新的信心向量评估。

警报系统： 设置异常情况警报（如网络断开、交易失败、达到亏损阈值）。

日志记录： 详细记录所有交易、API调用、LLM评估结果、RL模型状态，便于事后分析和调试。

4. 技术栈
编程语言： Python

数据库： InfluxDB/ClickHouse (时间序列), PostgreSQL (关系数据)

Web框架 (UI)： Streamlit / Dash

RL框架： Stable Baselines3 (PyTorch) / Ray RLlib

LLM框架： Hugging Face Transformers, Llama.cpp (用于本地部署优化)

搜索API封装： SerpApi, Google Custom Search API

交易所API SDK： python-binance

容器化： Docker (用于模块部署和环境隔离)

云服务： AWS/Azure/GCP (如果需要大规模计算资源和GPU)

5. 挑战与风险
LLM生成特征的准确性与鲁棒性： LLM对新闻的理解和情绪/信心提取的质量直接影响RL模型的表现，需要持续优化LLM的微调和提示词工程。

Mamba架构在金融RL中的适用性验证： Mamba是前沿技术，其在具体金融任务中的表现和稳定性需要大量实验验证。

历史新闻数据可得性： 外部搜索API对特定加密货币和久远日期的历史新闻覆盖度可能存在不足，可能需要整合多个来源。

过拟合风险： RL模型在训练数据上表现优异，但在未见过的真实市场中可能失效。需要严格的回测和验证流程，并考虑领域泛化技术。

交易成本： 频繁的日间交易会产生手续费和资金费率，必须在奖励函数中充分考虑，并优化交易频率。

市场动态变化： 加密货币市场波动性高，监管政策和市场情绪变化迅速，模型需要能够适应快速变化的市场环境。

6. 项目里程碑 (示例)
阶段1 (数据与基础架构)：✅ **COMPLETED**

- ✅ 市场数据获取与存储系统搭建
- ✅ Docker容器化开发环境部署
- ✅ InfluxDB时间序列数据库集成
- ✅ PostgreSQL关系数据库集成
- ✅ Binance API客户端实现（含速率限制）
- ✅ 历史OHLCV数据收集器
- ✅ 实时数据收集能力
- ✅ 项目结构和配置管理

阶段2 (LLM增强与数据融合)：✅ **COMPLETED**

- ✅ LLM模型选型与配置
- ✅ 7维信心向量系统设计
- ✅ RAG管道开发
- ✅ Mamba架构集成
- ✅ 数据融合引擎
- ✅ 强化学习环境构建
- ✅ 交易执行系统
- ✅ 风险管理系统

阶段2 (LLM增强与数据融合)：

LLM模型选型、本地部署与微调。

LLM-RAG流程开发，批量生成历史信心向量数据。

数据融合模块开发，形成增强数据集。

阶段3 (RL高级训练与Mamba探索)：

将LLM增强数据集成到RL观测空间。

初步尝试PPO/SAC等主流算法在增强数据集上训练。

研究Mamba架构在RL中的应用，并进行初步集成与实验。

阶段4 (回测与风险管理)：

开发完整的离线回测系统。

构建健壮的交易执行和风险管理模块。

基于回测结果进行策略优化和算法选择。

阶段5 (实盘部署与监控)：

构建实时监控仪表盘和警报系统。

小资金量实盘测试与持续优化。

7. 后续展望
探索更复杂的LLM应用，例如直接生成交易信号或更精细的事件影响分析。

集成更多非传统数据源，如链上数据、社交媒体情绪等。

开发自动化的策略迭代和模型更新机制。

扩展到更广阔的交易市场，如股票、外汇等。

8. Example for LLM prompt and Confidence Vector

## Environment Setup

### Using UV (Recommended)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "import cryptorl; print('Package installed successfully')"
```

### Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Docker Setup
```bash
# Build and run with docker-compose
docker-compose up --build

# Or run individual services
docker build -t cryptorl-agent .
docker run -it --env-file .env cryptorl-agent
```

9. Example for LLM prompt and Confidence Vector
{

  "role": "system",

  "content": "You are a market analysis expert (Athena).\n\nTask: After thoroughly searching and verifying the latest public data, news, and research, provide a short-term (next 1 month) investment confidence assessment for a single asset specified by the user.\n\nOutput format & rules:\n1. The **first line** must contain **only** a JSON array of four numbers—for example `[0.42,0.38,0.27,0.31]`—representing, in order, `[Fundamentals, Industry Condition, Geopolitics, Macroeconomics]`. Each value is in the range 0–1.\n2. The **second part** must explain the rationale for each score, including risks and opportunities.\n3. Any dimension > 0.5 implies a significant position increase; be especially cautious. Apart from the array and the explanation, output nothing else.\n4. Respond **only after** rigorous verification of up-to-date information. If data are insufficient, state explicitly: \"Insufficient up-to-date data to provide evaluation.\"\n5. All assessments must use a time horizon of exactly **one month**."

}

- 使用 uv 来配置环境和运行命令 (use uv to config env and use uv run)