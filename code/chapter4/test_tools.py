import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from tools import ToolExecutor, search
from dotenv import load_dotenv


class TestToolExecutor:
    """ToolExecutor类的单元测试"""

    def setup_method(self):
        """测试方法执行前的设置"""
        self.executor = ToolExecutor()


    def test_register_tool_adds_tool_correctly(self):
        """测试正确添加工具"""
        mock_func = Mock()
        self.executor.registerTool("TestTool", "测试工具描述", mock_func)

        assert "TestTool" in self.executor.tools
        assert self.executor.tools["TestTool"]["description"] == "测试工具描述"
        assert self.executor.tools["TestTool"]["func"] == mock_func


    def test_get_tool_returns_correct_function(self):
        """测试正确获取工具函数"""
        mock_func = Mock(return_value="执行结果")
        self.executor.registerTool("TestTool", "测试工具描述", mock_func)

        retrieved_func = self.executor.getTool("TestTool")

        assert retrieved_func == mock_func

        # 验证函数可以被调用
        result = retrieved_func("测试输入")
        assert result == "执行结果"


    def test_get_available_tools_formats_correctly(self):
        """测试正确格式化可用工具"""
        mock_func = Mock()
        self.executor.registerTool("TestTool", "测试工具描述", mock_func)
        self.executor.registerTool("AnotherTool", "另一个工具描述", mock_func)

        result = self.executor.getAvailableTools()

        assert "- TestTool: 测试工具描述" in result
        assert "- AnotherTool: 另一个工具描述" in result
        assert result.count('\n') == 1  # 两个工具之间有一个换行


class TestSearchFunction:
    """search函数的单元测试"""

    def setup_method(self):
        """测试方法执行前的设置"""
        # 保存原始环境变量
        self.original_api_key = os.environ.get("SERPAPI_API_KEY")

    def teardown_method(self):
        """测试方法执行后的清理"""
        # 恢复原始环境变量
        if self.original_api_key is not None:
            os.environ["SERPAPI_API_KEY"] = self.original_api_key
        elif "SERPAPI_API_KEY" in os.environ:
            del os.environ["SERPAPI_API_KEY"]

    @patch('tools.SerpApiClient')
    def test_search_returns_error_when_api_key_not_set(self, mock_serp_api_client):
        """测试API密钥未设置时返回错误"""
        # 确保环境变量未设置
        if "SERPAPI_API_KEY" in os.environ:
            del os.environ["SERPAPI_API_KEY"]

        result = search("测试查询")

        assert "错误：SERPAPI_API_KEY 未在 .env 文件中配置。" in result

    @patch('tools.SerpApiClient')
    def test_search_calls_api_client_with_correct_params(self, mock_serp_api_client):
        """测试使用正确参数调用API客户端"""
        # 设置模拟API密钥
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        # 模拟API客户端实例和返回结果
        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {
            "organic_results": [
                {"title": "测试结果", "snippet": "测试摘要", "link": "http://example.com"}
            ]
        }

        search("测试查询")

        # 验证API客户端被正确调用
        mock_serp_api_client.assert_called_once()
        args, kwargs = mock_serp_api_client.call_args
        params = args[0]  # 传递给SerpApiClient的第一个参数

        assert params["engine"] == "google"
        assert params["q"] == "测试查询"
        assert params["api_key"] == "test_api_key"
        assert params["gl"] == "cn"
        assert params["hl"] == "zh-cn"

    @patch('tools.SerpApiClient')
    def test_search_returns_answer_box_list_when_present(self, mock_serp_api_client):
        """测试当存在answer_box_list时返回该内容"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {
            "answer_box_list": ["直接答案1", "直接答案2"],
            "organic_results": [{"title": "其他结果", "snippet": "其他摘要"}]
        }

        result = search("测试查询")

        assert result == "直接答案1\n直接答案2"

    @patch('tools.SerpApiClient')
    def test_search_returns_answer_box_answer_when_present(self, mock_serp_api_client):
        """测试当存在answer_box时返回答案"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {
            "answer_box": {"answer": "直接答案"},
            "organic_results": [{"title": "其他结果", "snippet": "其他摘要"}]
        }

        result = search("测试查询")

        assert result == "直接答案"

    @patch('tools.SerpApiClient')
    def test_search_returns_knowledge_graph_description_when_present(self, mock_serp_api_client):
        """测试当存在知识图谱时返回描述"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {
            "knowledge_graph": {"description": "实体描述信息"},
            "organic_results": [{"title": "其他结果", "snippet": "其他摘要"}]
        }

        result = search("测试查询")

        assert result == "实体描述信息"

    @patch('tools.SerpApiClient')
    def test_search_returns_organic_results_when_other_answers_not_present(self, mock_serp_api_client):
        """测试当其他答案不存在时返回有机结果"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {
            "organic_results": [
                {"title": "结果1标题", "snippet": "结果1摘要"},
                {"title": "结果2标题", "snippet": "结果2摘要"},
                {"title": "结果3标题", "snippet": "结果3摘要"},
                {"title": "结果4标题", "snippet": "结果4摘要"}  # 应该只返回前3个
            ]
        }

        result = search("测试查询")

        # 验证只返回前3个结果
        assert "[1] 结果1标题" in result
        assert "[2] 结果2标题" in result
        assert "[3] 结果3标题" in result
        assert "[4] 结果4标题" not in result  # 第4个不应该出现

        # 验证格式正确
        assert "结果1摘要" in result
        assert "结果2摘要" in result
        assert "结果3摘要" in result

    @patch('tools.SerpApiClient')
    def test_search_returns_no_results_message_when_no_data(self, mock_serp_api_client):
        """测试当没有找到信息时返回相应消息"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {}

        result = search("测试查询")

        assert "对不起，没有找到关于 '测试查询' 的信息。" in result

    @patch('tools.SerpApiClient')
    def test_search_handles_exception_gracefully(self, mock_serp_api_client):
        """测试优雅处理异常"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        # 模拟API调用引发异常
        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.side_effect = Exception("API错误")

        result = search("测试查询")

        assert "搜索时发生错误: API错误" in result

    @patch('tools.SerpApiClient')
    def test_search_handles_missing_title_or_snippet(self, mock_serp_api_client):
        """测试处理缺失标题或摘要的情况"""
        os.environ["SERPAPI_API_KEY"] = "test_api_key"

        mock_client_instance = Mock()
        mock_serp_api_client.return_value = mock_client_instance
        mock_client_instance.get_dict.return_value = {
            "organic_results": [
                {"title": "有标题无摘要", "link": "http://example.com"},  # 缺少snippet
                {"snippet": "有摘要无标题", "link": "http://example.com"},  # 缺少title
                {"title": "完整结果", "snippet": "完整摘要"}  # 完整结果
            ]
        }

        result = search("测试查询")

        # 验证即使缺少某些字段也能正常处理
        assert "[1] 有标题无摘要" in result
        assert "[2] " in result  # 标题为空
        assert "有摘要无标题" in result
        assert "[3] 完整结果" in result
        assert "完整摘要" in result


class TestIntegrationToolExecutorWithSearch:
    """ToolExecutor与search函数集成测试"""

    def test_tool_executor_can_register_and_execute_search(self):
        """测试ToolExecutor可以注册和执行search函数"""
        executor = ToolExecutor()

        # 注册search函数
        executor.registerTool("Search", "搜索工具", search)

        # 获取并执行工具
        search_func = executor.getTool("Search")

        # 由于search函数需要API密钥，我们只验证函数是否正确获取
        assert search_func == search
        assert callable(search_func)


if __name__ == '__main__':
    pytest.main([__file__]) # 代表当前脚本的文件路径，只运行当前文件里的测试用例

