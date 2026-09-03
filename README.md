# 万象 · 生活智能助手（Wanxiang Life Assistant）

> ☀️ 天气 · 美食 · 出行 —— 基于 **LangGraph 多智能体 + MCP** 的生活服务问答助手。
> 前端以**树形对话**形式支持任意位置「编辑 / 重新生成 / 撤销」，后端以 **SSE 流式输出**实时回传执行过程。

> 🚧 **项目持续更新中**：本仓库会持续完善功能与模块，欢迎 Star / Issue 交流。

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [模块说明](#模块说明)
- [工作流程](#工作流程)
- [快速开始](#快速开始)
- [API 摘要](#api-摘要)
- [注意事项](#注意事项)
- [更新计划](#更新计划)

---

## 项目简介

「万象」是一个生活服务智能助手，用户可以用自然语言提问，例如：

- 天气类：「今天北京天气怎么样？要不要带伞？」
- 美食类：「周末想吃点家常菜，有什么推荐？红烧肉怎么做？」
- 出行类：「杭州出差坐高铁还是飞机？帮我查一下明天的高铁班次」

系统会自动**规划任务、调度多个子智能体（天气 / 美食 / 出行）并行或串行执行**，汇总后给出最终回答，并在回答前对内容做**安全审核与格式校验**。

## 功能特性

- **多智能体规划调度**：主图（supervisor）拆解用户问题为步骤图，分别派发给 weather / food / travel 三个子智能体，支持并行与依赖串行。
- **结构化输出**：子智能体通过 `response_format` 输出「结果、关键数据、执行状态」结构化字段，主图统一消费。
- **失败重试与重规划**：步骤失败或汇总未过验收时自动回到规划节点重新规划，带重试上限，避免死循环。
- **输出防护**：`guard` 节点对最终回答做内容安全（有害内容检测）与格式协议（JSON 合法性）双重校验，异常时降级重答。
- **SSE 流式输出**：规划步骤、工具结果、检查/反馈结果作为「过程」实时推送给前端（可折叠），最终结果作为 AI 消息展示。
- **树形对话**：任意用户消息可「编辑（分裂新分支）/ 撤销」，AI 消息可「重新生成」；分支之间可前后切换。
- **用户登录与会话管理**：登录（当前内置 `admin / admin`），会话密钥由后端生成并存入 Redis；支持历史会话列表、标题修改、切换回旧会话。
- **撤销持久化**：前端撤销消息会同步到后端 Redis，重新进入时历史记录不会错误渲染。
- **RAG 知识增强**：基于 Redis 向量库 + 内置气象标准 PDF 资料，可回答专业知识问题。

## 技术栈

| 端 | 技术 |
| --- | --- |
| 后端 | Python · FastAPI · LangChain 1.x · LangGraph · MCP (streamable-http) |
| 模型 | OpenAI 兼容云端模型（默认 DeepSeek）/ 本地 Ollama（Qwen）双通道 |
| 存储 | Redis（会话密钥、检查点持久化、撤销标记、向量库） |
| 工具 | FastMCP 本地服务（天气 / 菜谱）+ 第三方 MCP（12306、航班、菜谱等） |
| 前端 | Vue 3 · Vite · 原生 SSE（fetch ReadableStream） |

## 项目结构

```
project3/
├── backend/                  # 后端（FastAPI + LangGraph 主控）
│   ├── api.py                #   REST / SSE 接口：登录、会话、流式回答、历史、撤销
│   ├── agent.py              #   主图：plan→schedule→步骤→check→merge→feedback→guard
│   ├── model.py              #   模型初始化（云端 / Ollama 双通道，lru_cache 单例）
│   ├── rag.py                #   RAG：PDF 解析 → Redis 向量库写入 / 删除
│   ├── mcp.json              #   ⚠️ MCP 服务器真实配置（含 Key，不入库，见 mcp_example.json）
│   ├── mcp_example.json      #   可提交的 MCP 配置模板（Key 打码）
│   ├── subagent/             #   子智能体（独立可运行的 LangGraph 图）
│   │   ├── weather.py        #     天气：open-meteo + 空气质量 MCP
│   │   ├── food.py           #     美食：菜谱/家常菜 MCP
│   │   └── travel.py         #     出行：12306 / 航班 / 地图 MCP
│   └── rag/                  #   内置气象标准 / 气候 PDF 资料（RAG 语料）
├── tool/                     # 本地 MCP 工具服务器
│   ├── weather.py            #   天气服务（FastMCP，端口 8000）
│   ├── food.py               #   菜谱服务（FastMCP，端口 8002）
│   └── openapi-v1.yaml       #   第三方 MCP 接口定义
├── front/                    # 前端（Vue 3 + Vite）
│   ├── src/App.vue           #   聊天主界面：登录、树形对话、会话侧边栏、SSE 过程面板
│   ├── src/style.css         #   暖色「生活助手」主题样式
│   └── vite.config.js        #   dev 代理 → 后端 127.0.0.1:8080
├── .gitignore                # 忽略规则（密钥/缓存/构建产物）
└── requirements.txt          # Python 依赖清单
```

## 模块说明

### backend/api.py —— 对外接口层
FastAPI 应用，负责鉴权、会话与 HTTP/SSE 适配：

| 接口 | 说明 |
| --- | --- |
| `POST /login` | 用户登录，当前内置账号 `admin / admin` |
| `GET /sessions` | 列出当前用户全部历史会话（标题/创建时间/最近活动） |
| `POST /sessions/rename` | 修改会话标题 |
| `POST /answer` | 同步回答（返回结果与前后检查点 id） |
| `POST /answer/stream` | SSE 流式回答：`process` 事件推送执行过程，`done` 事件携带最终结果与新会话密钥 |
| `GET /history` | 重建会话可视历史（过滤已撤销消息） |
| `POST /revoke` | 前端撤销消息时同步到 Redis，防止重新进入时错误渲染 |

新对话时，后端在**第一条用户消息**随机生成会话密钥并回传；所有会话密钥、撤销标记持久化在 Redis。

### backend/agent.py —— 多智能体主图
LangGraph 状态图，节点包括：

- `input` / `plan`：意图解析与步骤规划（输出步骤图，互不依赖的步骤并行调度）
- `schedule`：将步骤分发给 weather / food / travel 子图（`Send` 并行 + `depends_on` 串行）
- `weather` / `food` / `travel`：调用子智能体，读取其结构化输出（`result / key_data / status`）
- `check`：按子图返回的 `status`（success / no_data / failed）判定，失败则带原因回 `plan` 重规划（有重试上限）
- `merge`：汇总各步骤结论
- `feedback`：验收汇总结果，不通过则结合反馈重答
- `guard`：最终输出防护（内容安全审核 + 格式/协议校验）
- `fallback`：无法识别任务时的兜底回复

### backend/subagent/ —— 子智能体
每个子智能体是独立的 LangGraph 图，通过 `create_agent` 的 `response_format` 强制输出：

```python
class AgentOutput(TypedDict):
    result: str    # 面向用户的最终自然语言回复
    key_data: dict # 关键数据（气温、班次、步骤等结构化字段）
    status: str    # 执行状态：success / no_data / failed
```

### tool/ —— 本地 MCP 工具服务器
基于 FastMCP 的本地服务，`start.bat` 依次拉起：

- `weather.py`（端口 8000）：open-meteo 天气 + 时间
- `food.py`（端口 8002）：TheMealDB 菜谱

### front/ —— 前端
Vue 3 单页应用：

- **登录页**：账号密码登录（`admin / admin`）
- **树形对话**：消息以分支树组织，用户消息可编辑（分裂新分支）、撤销；AI 消息可重新生成、不允许撤销
- **思考过程面板**：SSE 过程中实时折叠展示「规划步骤 / 工具结果 / 检查 / 反馈」，结果经整理后展示
- **我的会话**：历史会话列表、标题重命名、一键切换旧会话

## 工作流程

```
用户提问
  └→ plan（规划步骤图）
       └→ schedule（派发，并行/串行）
            ├→ weather / food / travel 子智能体（结构化输出）
            └→ check（按状态判定，失败→重规划，带重试上限）
                 └→ merge（汇总）
                      └→ feedback（验收，不通过→结合反馈重答）
                           └→ guard（内容安全 + 格式协议校验）
                                └→ 最终结果（SSE done → AI 消息）
```

## 快速开始

### 环境要求
- Python 3.10+
- Redis（默认 `redis://localhost:26379`，可通过 `REDIS_URL` 覆盖）
- （可选）本地 Ollama，用于本地模型通道
- Node.js 18+（前端）

### 1. 配置
复制配置模板并按需填写（真实 `.env` 已忽略，不入库）：

```bash
# backend/ 下创建 .env，参考模板：
REDIS_URL = 'redis://localhost:26379'

# 云端模型（默认）
model = 'deepseek-v4-flash'
model_provider = 'openai'
model_api = 'sk-xxx'                       # 你的 API Key
base_url = 'https://opencode.ai/zen/go/v1'

# 本地 Ollama 通道（route_model / chat_model 设为 local 时生效）
ollama_model = 'qwen3.8:27b'
ollama_url = 'http://localhost:11434'
```

MCP 配置：将 `backend/mcp_example.json` 复制为 `backend/mcp.json`（真实端点，含 Key，不入库）并填写。

### 2. 安装依赖
```bash
pip install -r requirements.txt
cd front && npm install
```

### 3. 启动
```bash
# 一键启动（Windows powershell/cmd）：本地 MCP → 后端 → 前端
start.bat
```

或手动启动：

```bash
# 本地 MCP 工具
cd tool && python weather.py
cd tool && set food_port=8002 && python food.py

# 后端
cd backend && python -m uvicorn api:app --host 127.0.0.1 --port 8080

# 前端（dev，代理 /answer、/history 等到 8080）
cd front && npm run dev
```

访问 `http://127.0.0.1:5173`，登录账号 `admin`，密码 `admin`。

## API 摘要

见 [backend/api.py 模块说明](#backendapi.py--对外接口层)；流式回答 SSE 事件：

- `event: process` → `{ id, node, label, type, status, title, content, key_data, steps }`
- `event: done` → `{ result, session_key, human_message_id, ai_message_id, before_checkpoint_id, after_checkpoint_id }`
- `event: error` → `{ message }`

## 注意事项

- `backend/mcp.json`、`backend/.env`、`tool/.env` 含密钥，已由 `.gitignore` 显式忽略，**切勿提交**；远端请基于 `mcp_example.json` 与 `.env.example` 模板自建。
- 登录为演示实现（账号写死后端），生产环境请替换为真实鉴权。
- `backend/rag/` 中的 PDF 为内置气象资料语料（RAG 依赖）；如需精简仓库体积可调整后自行删除。

## 更新计划

- [ ] 多轮规划上下文增强（结合历史问题的重规划）
- [ ] 真实用户系统（注册 / Token / 权限）
- [ ] 更多生活领域子智能体（学习、医疗、运动等）
- [ ] 前端主题与移动端适配
- [ ] 单元测试与 CI

---

**持续更新中** —— 欢迎在 [Issues](https://github.com/) 提出建议与意见。