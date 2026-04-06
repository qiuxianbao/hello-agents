"""
AutoGen 软件开发团队协作案例
在本案例中，软件开发的流程是相对固定的（需求->编码->审查->测试），因此 RoundRobinGroupChat (轮询群聊)是理想的选择。
"""

import asyncio
import os

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 先测试一个版本，使用 OpenAI 客户端
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console


def create_openai_model_client():
    """创建 OpenAI 模型客户端用于测试"""
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )


def create_product_manager(model_client):
    """创建产品经理智能体"""
    system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应，并在分析完成后说"请工程师开始实现"。"""

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )


def create_engineer(model_client):
    """创建软件工程师智能体"""
    system_message = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

你的技术专长包括：
1. **Python 编程**：熟练掌握 Python 语法和最佳实践
2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

请提供完整的可运行代码，并在完成后说"请代码审查员检查"。"""

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )


def create_code_reviewer(model_client):
    """创建代码审查员智能体"""
    system_message = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。"""

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )


def create_user_proxy():
    """创建用户代理智能体
    代表最终用户，发起初始任务，并负责执行和验证最终交付的代码

    """
    return UserProxyAgent(
        name="UserProxy",
        description="""用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 执行最终的代码实现
3. 验证功能是否符合预期
4. 提供用户反馈和建议

完成测试后请回复 TERMINATE。""",
    )


async def run_software_development_team():
    """运行软件开发团队协作"""

    print("🔧 正在初始化模型客户端...")

    # 先使用标准的 OpenAI 客户端测试
    model_client = create_openai_model_client()

    print("👥 正在创建智能体团队...")

    # 创建智能体团队
    """
    AssistantAgent(助理智能体)： 这是任务的主要解决者，其核心是封装了一个大型语言模型（LLM）
    
    UserProxyAgent(用户代理智能体)：它扮演着双重角色,既是人类用户的“代言人”，负责发起任务和传达意图；
    又是一个可靠的“执行器”，可以配置为执行代码或调用工具，并将结果反馈给其他智能体。
    这种设计清晰地区分了“思考”（由 AssistantAgent 完成）与“行动”。 
    """
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()

    # 添加终止条件
    termination = TextMentionTermination("TERMINATE")

    # 创建团队聊天
    """
    轮询群聊 (RoundRobinGroupChat)： 
    这是一种明确的、顺序化的对话协调机制。它会让参与的智能体按照预定义的顺序依次发言。
    
    场景：
    这种模式非常适用于流程固定的任务。
    
    """
    team_chat = RoundRobinGroupChat(
        participants=[  # participants 列表的顺序决定了智能体发言的先后次序
            product_manager,
            engineer,
            code_reviewer,
            user_proxy
        ],
        termination_condition=termination,  # 是控制协作流程何时结束的关键
        max_turns=20,  # 增加最大轮次，用于防止对话陷入无限循环
    )

    # 定义开发任务
    task = """我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。"""

    # 执行团队协作
    print("🚀 启动 AutoGen 软件开发团队协作...")
    print("=" * 60)

    # 使用 Console 来显示对话过程
    """
    🔧 正在初始化模型客户端...
    👥 正在创建智能体团队...
    🚀 启动 AutoGen 软件开发团队协作...
    ============================================================
    ---------- TextMessage (user) ----------
    我们需要开发一个比特币价格显示应用，具体要求如下：
    ...
    请团队协作完成这个任务，从需求分析到最终实现。
    ---------- TextMessage (ProductManager) ----------
    ### 1. 需求理解与分析
    ...
    请工程师开始实现。
    ---------- TextMessage (Engineer) ----------
    ### 技术方案实施
    ...
    请代码审查员检查。
    ---------- TextMessage (CodeReviewer) ----------
    ### 代码审查
    ...
    代码审查完成，请用户代理测试。
    ---------- TextMessage (UserProxy) ----------
    已经完成需求
    ---------- TextMessage (ProductManager) ----------
    太好了，感谢您的反馈！如果在使用过程中有任何问题，或者有其他功能需求和改进建议，请随时告知我们。我们会持
    续提供支持和改进。期待您对我们的应用
    有愉快的使用体验！
    ---------- TextMessage (Engineer) ----------
    很高兴听到项目顺利完成。如果您或用户有任何问题或者需要帮助，请随时联系我们。感谢您对我们工作的支持，让我
    们一起确保应用稳定运行并不断优化用户
    体验！
    ---------- TextMessage (CodeReviewer) ----------
    非常感谢大家的努力与协作，使得项目能够顺利完成。未来若有更多技术支持的需求或者需要改进的地方，我们愿意为
    项目的持续优化贡
    献力量。期待用户能够享受到流畅的体验，同时也欢迎提出更多的反馈与建议。再次感谢团队的合作！
    ---------- TextMessage (UserProxy) ----------
    Enter your response: TERMINATE
    ============================================================
    ✅ 团队协作完成！
    📋 协作结果摘要：
    - 参与智能体数量：4个
    - 任务完成状态：成功
    
    """
    result = await Console(team_chat.run_stream(task=task))

    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")

    return result


# 主程序入口
if __name__ == "__main__":
    try:
        # 运行异步协作流程
        result = asyncio.run(run_software_development_team())

        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")

    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env 文件中的配置是否正确")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback

        traceback.print_exc()
