# 记忆与RAG系统整体架构

```
pip install "hello-agents[all]==0.2.0"
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm

HelloAgents记忆系统-四层架构
├── 基础设施层 (Infrastructure Layer)
│ ├── MemoryManager - 记忆管理器（统一调度和协调）
│ ├── MemoryItem - 记忆数据结构（标准化记忆项）
│ ├── MemoryConfig - 配置管理（系统参数设置）
│ └── BaseMemory - 记忆基类（通用接口定义）
├── 记忆类型层 (Memory Types Layer)
│ ├── WorkingMemory - 工作记忆（临时信息，TTL管理）
│ ├── EpisodicMemory - 情景记忆（具体事件，时间序列）
│ ├── SemanticMemory - 语义记忆（抽象知识，图谱关系）
│ └── PerceptualMemory - 感知记忆（多模态数据）
├── 存储后端层 (Storage Backend Layer)
│ ├── QdrantVectorStore - 向量存储（高性能语义检索）
│ ├── Neo4jGraphStore - 图存储（知识图谱管理）
│ └── SQLiteDocumentStore - 文档存储（结构化持久化）
└── 嵌入服务层 (Embedding Service Layer)
├── DashScopeEmbedding - 通义千问嵌入（云端API）
├── LocalTransformerEmbedding - 本地嵌入（离线部署）
└── TFIDFEmbedding - TFIDF嵌入（轻量级兜底）

hello-agents/
├── hello_agents/
│ ├── memory/ # 记忆系统模块
│ │ ├── base.py # 基础数据结构（MemoryItem, MemoryConfig, BaseMemory）
│ │ ├── manager.py # 记忆管理器（统一协调调度） -- 统一协调调度
│ │ ├── embedding.py # 统一嵌入服务（DashScope/Local/TFIDF）
│ │ ├── types/ # 记忆类型实现
│ │ │ ├── working.py # 工作记忆（TTL管理，纯内存）
│ │ │ ├── episodic.py # 情景记忆（事件序列，SQLite+Qdrant）
│ │ │ ├── semantic.py # 语义记忆（知识图谱，Qdrant+Neo4j）
│ │ │ └── perceptual.py # 感知记忆（多模态，SQLite+Qdrant）
│ │ ├── storage/ # 存储后端实现
│ │ │ ├── qdrant_store.py # Qdrant向量存储（高性能向量检索）
│ │ │ ├── neo4j_store.py # Neo4j图存储（知识图谱管理）
│ │ │ └── document_store.py # SQLite文档存储（结构化持久化）
│ │ └── rag/ # RAG系统
│ │ ├── pipeline.py # RAG管道（端到端处理）
│ │ └── document.py # 文档处理器（多格式解析）
│ └── tools/builtin/ # 扩展内置工具
│       ├── memory_tool.py # 记忆工具（Agent记忆能力） -- 对外
│       └── rag_tool.py # RAG工具（智能问答能力）
└──
```


```
（1）什么是RAG？
检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了信息检索和文本生成的技术。
它的核心思想是：在生成回答之前，先从外部知识库中检索相关信息，然后将检索到的信息作为上下文提供给大语言模型，从而生成更准确、更可靠的回答。
因此，检索增强生成可以拆分为三个词汇。
【检索】是指从知识库中查询相关内容；
【增强】是将检索结果融入提示词，辅助模型生成；
【生成】则输出兼具准确性与透明度的答案。

（2）基本工作流程
一个完整的RAG应用流程主要分为两大核心环节。
在数据准备阶段，系统通过数据提取、文本分割和向量化，将外部知识构建成一个可检索的数据库。
随后在应用阶段，系统会响应用户的提问，从数据库中检索相关信息，将其注入Prompt，并最终驱动大语言模型生成答案。

HelloAgents RAG系统
├── 文档处理层 (Document Processing Layer)
│ ├── DocumentProcessor - 文档处理器（多格式解析）
│ ├── Document - 文档对象（元数据管理）
│ └── Pipeline - RAG管道（端到端处理）
├── 嵌入表示层 (Embedding Layer)
│ └── 统一嵌入接口 - 复用记忆系统的嵌入服务
├── 向量存储层 (Vector Storage Layer)
│ └── QdrantVectorStore - 向量数据库（命名空间隔离）
└── 智能问答层 (Intelligent Q&A Layer)
├── 多策略检索 - 向量检索 + MQE + HyDE
├── 上下文构建 - 智能片段合并与截断
└── LLM增强生成 - 基于上下文的准确问答
```

