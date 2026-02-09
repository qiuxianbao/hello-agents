"""
CodebaseMaintainer - 代码库维护助手

# 长智能体的关键特性
* 上下文管理，ContextBuilder 确保每次对话都有高质量的上下文，自动汇集相关笔记(特别是 blocker 类型)
* 跨会话的连贯性，智能体通过 NoteTool 保持了跨多天、多个会话的任务连贯性
* 即时的文件系统访问，TerminalTool 支持灵活的代码探索，无需预先索引整个代码库，可以即时查看具体文件内容，支持复杂的文本处理(grep、awk等)。
* 自动化的知识管理，系统自动化地管理发现的知识，发现问题时自动创建 blocker 笔记，讨论计划时自动创建 action 笔记，关键信息自动存储到记忆系统。


# 业务场景
假设我们正在维护一个中型 Python Web 应用，这个代码库包含约 50 个 Python 文件，使用 Flask 框架构建，涵盖数据模型、业务逻辑、API 接口等多个模块，同时存在一些技术债务需要逐步清理。
在这样的场景下，我们需要一个智能助手来帮助我们
1.探索代码库，理解项目结构、依赖关系和代码风格；
2.识别代码中的问题，比如代码重复、复杂度过高、缺少测试等；
3.追踪任务进度，记录待办事项、已完成工作和遇到的阻塞；
4.并基于历史上下文提供连贯的重构建议。

# 挑战与解决方案
这个场景面临几个典型的长程任务挑战。
首先是信息量超出上下文窗口的问题，整个代码库可能包含数万行代码，无法一次性放入上下文窗口，我们通过使用 TerminalTool 进行即时、按需的代码探索来解决这个问题，只在需要时查看具体文件。
其次是跨会话的状态管理挑战，重构任务可能持续数天，需要跨多个会话保持进度，我们使用 NoteTool记录阶段性进展、待办事项和关键决策来应对。
最后是上下文质量与相关性的问题，每次对话需要回顾相关的历史信息，但不能被无关信息淹没，我们通过 ContextBuilder 智能筛选和组织上下文，确保高信号密度。


完整的长程智能体实现，整合:
1. ContextBuilder - 上下文管理
2. NoteTool - 结构化笔记
3. TerminalTool - 即时文件访问
4. MemoryTool - 对话记忆

关键改进：使用 Agentic 方式，让 agent 自主决定使用哪些工具
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from hello_agents import HelloAgentsLLM
from hello_agents.agents import FunctionCallAgent
from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.tools import MemoryTool, NoteTool, TerminalTool
from hello_agents.tools.registry import ToolRegistry
from hello_agents.core.message import Message


class CodebaseMaintainer:
    """代码库维护助手 - 长程智能体示例

    整合 ContextBuilder + NoteTool + TerminalTool + MemoryTool
    实现跨会话的代码库维护任务管理
    
    核心特性：
    - Agent 自主使用工具探索代码库
    - 不预定义工作流，完全基于 agent 决策
    - 跨会话记忆和上下文管理
    """

    def __init__(
        self,
        project_name: str,
        codebase_path: str,
        llm: Optional[HelloAgentsLLM] = None
    ):
        self.project_name = project_name
        self.codebase_path = codebase_path
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化 LLM
        self.llm = llm or HelloAgentsLLM()

        # 初始化工具
        self.memory_tool = MemoryTool(
            user_id=project_name,
            memory_types=["working"]
        )
        self.note_tool = NoteTool(workspace=f"./{project_name}_notes")
        self.terminal_tool = TerminalTool(workspace=codebase_path, timeout=60)

        # 初始化上下文构建器
        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=None,  # 本案例不使用 RAG
            config=ContextConfig(
                max_tokens=4000,
                reserve_ratio=0.15,
                min_relevance=0.2,
                enable_compression=True
            )
        )

        # 创建工具注册表并注册工具
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(self.terminal_tool)
        self.tool_registry.register_tool(self.note_tool)
        self.tool_registry.register_tool(self.memory_tool)

        # 创建 Agent
        self.agent = FunctionCallAgent(
            name="CodebaseMaintainer",
            llm=self.llm,
            system_prompt=self._build_base_system_prompt(),
            tool_registry=self.tool_registry,
            enable_tool_calling=True,
            max_tool_iterations=30
        )

        # 对话历史
        self.conversation_history: List[Message] = []

        # 统计信息
        self.stats = {
            "session_start": datetime.now(),
            "commands_executed": 0,
            "notes_created": 0,
            "issues_found": 0,
            "tool_calls": 0
        }

        print(f"✅ 代码库维护助手已初始化: {project_name} (Agentic Mode)")
        print(f"📁 工作目录: {codebase_path}")
        print(f"🆔 会话ID: {self.session_id}")
        print(f"🔧 可用工具: {', '.join(self.tool_registry.list_tools())}")

    def run(self, user_input: str, mode: str = "auto") -> str:
        """运行助手（Agentic 方式）

        Args:
            user_input: 用户输入
            mode: 运行模式提示（给 agent 提供方向性建议）
                - "auto": 自动决策是否使用工具
                - "explore": 建议 agent 侧重代码探索
                - "analyze": 建议 agent 侧重问题分析
                - "plan": 建议 agent 侧重任务规划

        Returns:
            str: 助手的回答
        """
        print(f"\n{'='*80}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*80}\n")

        # 第一步: 检索相关笔记（为 agent 提供上下文）
        relevant_notes = self._retrieve_relevant_notes(user_input)
        note_packets = self._notes_to_packets(relevant_notes)

        # 第二步: 【构建优化的上下文】
        context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self._build_system_instructions(mode),  # 根据模式去创建提示词
            additional_packets=note_packets
        )

        # 第三步: 让 Agent 自主决策和使用工具
        print("🤖 Agent 正在思考并决定使用哪些工具...\n")
        
        # 更新 agent 的系统提示（包含上下文）
        self.agent.system_prompt = context
        
        # 调用 agent（agent 会自主决定是否使用工具）
        response = self.agent.run(user_input)

        # 第四步: 统计工具使用情况
        self._track_tool_usage()

        # 第五步: 更新对话历史
        self._update_history(user_input, response)

        print(f"\n🤖 助手: {response}\n")
        print(f"{'='*80}\n")

        return response

    def _build_base_system_prompt(self) -> str:
        """构建基础系统提示"""
        return f"""你是 {self.project_name} 项目的代码库维护助手。

你的核心能力:
1. 使用 TerminalTool 探索代码库
   - 你可以执行任何 shell 命令: ls, cat, grep, find, git 等
   - 工作目录: {self.codebase_path}
   
2. 使用 NoteTool 记录发现和任务
   - 创建笔记记录重要发现
   - 笔记类型: blocker(阻塞问题)、action(行动计划)、task_state(任务状态)、conclusion(结论)
   
3. 使用 MemoryTool 存储关键信息
   - 记住重要的上下文信息
   - 跨会话保持连贯性

当前会话ID: {self.session_id}

重要原则:
- 你要自主决定使用哪些工具、执行什么命令
- 探索代码库时，先了解整体结构，再深入细节
- 发现重要信息时，主动使用 NoteTool 记录
- 保持回答的专业性和实用性
"""

    def _track_tool_usage(self):
        """统计工具使用情况"""
        # 从 agent 的执行历史中统计
        if hasattr(self.agent, 'message_history'):
            for msg in self.agent.message_history[-10:]:  # 只看最近10条
                if msg.role == "tool":
                    self.stats["tool_calls"] += 1
                    # 根据工具名统计
                    if "terminal" in str(msg.content).lower() or "command" in str(msg.content).lower():
                        self.stats["commands_executed"] += 1
                    elif "note" in str(msg.content).lower():
                        if "create" in str(msg.content).lower():
                            self.stats["notes_created"] += 1

    def _retrieve_relevant_notes(self, query: str, limit: int = 3) -> List[Dict]:
        """检索相关笔记"""
        try:
            # 优先检索 blocker
            blockers_raw = self.note_tool.run({
                "action": "list",
                "note_type": "blocker",
                "limit": 2
            })
            blockers = self._normalize_note_results(blockers_raw)

            # 搜索相关笔记
            search_results_raw = self.note_tool.run({
                "action": "search",
                "query": query,
                "limit": limit
            })
            search_results = self._normalize_note_results(search_results_raw)

            # 合并去重
            all_notes = {}
            for note in blockers + search_results:
                if not isinstance(note, dict):
                    continue
                note_id = note.get('note_id') or note.get('id')
                if not note_id:
                    continue
                if note_id not in all_notes:
                    all_notes[note_id] = note

            return list(all_notes.values())[:limit]

        except Exception as e:
            print(f"[WARNING] 笔记检索失败: {e}")
            return []

    def _normalize_note_results(self, result: Any) -> List[Dict]:
        """将笔记工具的返回值转换为笔记字典列表"""
        if not result:
            return []

        if isinstance(result, dict):
            return [result]

        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]

        if isinstance(result, str):
            text = result.strip()
            if not text:
                return []
            if text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                    return self._normalize_note_results(parsed)
                except Exception:
                    return []
            return []

        return []

    def _notes_to_packets(self, notes: List[Dict]) -> List[ContextPacket]:
        """将笔记转换为上下文包"""
        packets = []

        for note in notes:
            if not isinstance(note, dict):
                continue
            # 根据笔记类型设置不同的相关性分数
            relevance_map = {
                "blocker": 0.9,
                "action": 0.8,
                "task_state": 0.75,
                "conclusion": 0.7
            }

            note_type = note.get('type', 'general')
            relevance = relevance_map.get(note_type, 0.6)

            content = f"[笔记:{note.get('title', 'Untitled')}]\n类型: {note_type}\n\n{note.get('content', '')}"
            updated_at = note.get('updated_at')
            try:
                note_timestamp = datetime.fromisoformat(updated_at) if updated_at else datetime.now()
            except (ValueError, TypeError):
                note_timestamp = datetime.now()

            packets.append(ContextPacket(
                content=content,
                timestamp=note_timestamp,
                token_count=len(content) // 4,
                relevance_score=relevance,
                metadata={
                    "type": "note",
                    "note_type": note_type,
                    "note_id": note.get('note_id') or note.get('id')
                }
            ))

        return packets

    def _build_system_instructions(self, mode: str) -> str:
        """构建系统指令（Agentic 方式）"""
        base_instructions = self._build_base_system_prompt()

        mode_hints = {
            "explore": """
用户当前关注: 探索代码库

建议策略:
- 考虑使用 TerminalTool 了解代码结构（如 find, ls, tree）
- 查看关键文件（如 README, 主要模块）
- 将架构信息记录到笔记方便后续查阅
""",
            "analyze": """
用户当前关注: 分析代码质量

建议策略:
- 考虑使用 grep 查找潜在问题（TODO, FIXME, BUG）
- 分析代码复杂度和结构
- 将发现的问题记录为 blocker 或 action 笔记
""",
            "plan": """
用户当前关注: 任务规划

建议策略:
- 回顾历史笔记了解当前进度
- 基于已有信息制定行动计划
- 创建或更新 task_state 类型的笔记
""",
            "auto": """
用户当前关注: 自由对话

建议策略:
- 根据用户需求灵活决策
- 在需要时主动使用工具获取信息
- 不需要时可以直接回答
"""
        }

        return base_instructions + "\n" + mode_hints.get(mode, mode_hints["auto"])


    def _update_history(self, user_input: str, response: str):
        """更新对话历史"""
        self.conversation_history.append(
            Message(content=user_input, role="user", timestamp=datetime.now())
        )
        self.conversation_history.append(
            Message(content=response, role="assistant", timestamp=datetime.now())
        )

        # 限制历史长度(保留最近10轮对话)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    # === 便捷方法 ===

    def explore(self, target: str = ".") -> str:
        """探索代码库（Agentic 方式）
        
        Agent 会自主决定使用哪些命令来探索代码库
        """
        return self.run(f"请探索 {target} 的代码结构，了解项目组织方式", mode="explore")

    def analyze(self, focus: str = "") -> str:
        """分析代码质量（Agentic 方式）
        
        Agent 会自主决定如何分析代码质量
        """
        query = f"请分析代码质量" + (f"，重点关注{focus}" if focus else "")
        return self.run(query, mode="analyze")

    def plan_next_steps(self) -> str:
        """规划下一步任务（Agentic 方式）
        
        Agent 会查看历史笔记并规划下一步
        """
        return self.run("根据我们之前的分析和当前进度，规划下一步任务", mode="plan")

    def execute_command(self, command: str) -> str:
        """执行终端命令"""
        result = self.terminal_tool.run({"command": command})
        self.stats["commands_executed"] += 1
        return result

    def create_note(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: List[str] = None
    ) -> str:
        """创建笔记"""
        result = self.note_tool.run({
            "action": "create",
            "title": title,
            "content": content,
            "note_type": note_type,
            "tags": tags or [self.project_name]
        })
        self.stats["notes_created"] += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        # 获取笔记摘要
        try:
            note_summary = self.note_tool.run({"action": "summary"})
        except:
            note_summary = {}

        return {
            "session_info": {
                "session_id": self.session_id,
                "project": self.project_name,
                "duration_seconds": duration
            },
            "activity": {
                "commands_executed": self.stats["commands_executed"],
                "notes_created": self.stats["notes_created"],
                "issues_found": self.stats["issues_found"]
            },
            "notes": note_summary
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成会话报告"""
        report = self.get_stats()

        if save_to_file:
            report_file = f"maintainer_report_{self.session_id}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file
            print(f"📄 报告已保存: {report_file}")

        return report


def main():
    """主函数 - 演示 CodebaseMaintainer 的使用（Agentic 版本）
    
    在这个版本中：
    - Agent 自主决定使用哪些工具
    - 不预定义工作流
    - Agent 根据需求灵活探索代码库
    """
    print("=" * 80)
    print("CodebaseMaintainer 演示（Agentic 版本）")
    print("=" * 80 + "\n")

    # 初始化助手
    maintainer = CodebaseMaintainer(
        project_name="my_flask_app",
        codebase_path="./my_flask_app",
        llm=HelloAgentsLLM()
    )

    # 探索代码库（Agent 自主决定如何探索）
    print("\n### 探索代码库（Agent 自主探索）###")
    response = maintainer.explore()

    # 分析代码质量（Agent 自主决定分析方法）
    print("\n### 分析代码质量（Agent 自主分析）###")
    response = maintainer.analyze()

    # 规划下一步（Agent 基于历史信息规划）
    print("\n### 规划下一步任务（Agent 自主规划）###")
    response = maintainer.plan_next_steps()

    # 生成报告
    print("\n### 生成会话报告 ###")
    report = maintainer.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
