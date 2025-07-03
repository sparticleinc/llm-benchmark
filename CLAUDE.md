# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

LLM-Benchmark 是一个综合性的 LLM 性能测试工具，支持自动化压力测试和性能报告生成。该项目旨在对 OpenAI 兼容的 API 端点进行基准测试，支持多种并发级别和详细的指标收集。

## 关键命令

### 环境设置
```bash
# 创建虚拟环境（Windows）
python -m venv venv
.\venv\Scripts\activate

# 创建虚拟环境（Linux/macOS）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行测试

#### 全自动性能测试（推荐）
```bash
python run_benchmarks.py --llm_url "http://localhost:8091/v1" --api_key "your-key" --model "gbase-llama-33" --adaptive
```

#### 单轮测试
```bash
python llm_benchmark.py --llm_url "http://localhost:8091/v1" --api_key "your-key" --model "gbase-llama-33" --num_requests 100 --concurrency 10
```

## 架构与代码结构

### 核心组件

1. **run_benchmarks.py**：高级编排脚本
   - 管理不同并发级别的多轮测试
   - 实现自适应模式，自动增加并发数直到性能下降
   - 使用 matplotlib 生成性能可视化报告
   - 处理结果聚合和汇总统计

2. **llm_benchmark.py**：核心基准测试引擎
   - 使用 asyncio 和 OpenAI AsyncClient 实现异步并发请求处理
   - 支持流式响应，跟踪首令牌时间（TTFT）
   - 全面的错误分类（timeout、rate_limit、auth_error、network_error 等）
   - 详细的指标收集：延迟、吞吐量、令牌生成速率

### 关键设计模式

- **异步/并发架构**：使用 asyncio 和 Semaphore 实现受控并发
- **基于队列的工作者模式**：工作者从队列中拉取任务进行负载分配
- **流式响应处理**：正确处理 OpenAI 流式 API 并进行令牌计数
- **错误恢复能力**：全面的错误处理，包含分类和样本收集

### 重要实现细节

- 该工具使用 OpenAI 的 AsyncOpenAI 客户端，兼容任何 OpenAI 兼容的 API
- 支持短提示和长上下文测试（通过 --use_long_context 标志控制）
- 自适应模式基于成功率阈值（默认 95%）动态调整并发数
- 所有测试结果都保存为带时间戳的 JSON 文件，便于后续分析
- 性能图表自动生成并保存为 PNG 格式

## 开发指南

- 修改测试配置时，需同时更新 run_benchmarks.py 中的 `configurations` 列表，并确保与自适应模式兼容
- 错误分类逻辑位于 `make_request` 函数中 - 添加新错误类型时保持一致性
- `process_stream` 中的流式处理器必须同时处理 content 和 reasoning_content 字段，以兼容不同模型
- 保持日志信息丰富但不冗长 - 使用 logging.debug 记录详细跟踪信息