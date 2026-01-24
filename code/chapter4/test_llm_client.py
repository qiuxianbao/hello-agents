import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from dotenv import load_dotenv

from llm_client import HelloAgentsLLM


class TestDotenv:

    def test_load_env(self):
        """测试dotenv文件是否正确加载"""
        load_dotenv()
        print(os.getenv("SERPAPI_API_KEY"))


class TestHelloAgentsLLM:
    """HelloAgentsLLM类的单元测试"""

    def setup_method(self):
        """测试方法执行前的设置"""
        # 保存原始环境变量
        self.original_model_id = os.environ.get("LLM_MODEL_ID")
        self.original_api_key = os.environ.get("LLM_API_KEY")
        self.original_base_url = os.environ.get("LLM_BASE_URL")
        self.original_timeout = os.environ.get("LLM_TIMEOUT")

        # 设置测试用的环境变量
        os.environ["LLM_MODEL_ID"] = "test-model"
        os.environ["LLM_API_KEY"] = "test-api-key"
        os.environ["LLM_BASE_URL"] = "https://test.base.url"

    def teardown_method(self):
        """测试方法执行后的清理"""
        # 恢复原始环境变量
        if self.original_model_id is not None:
            os.environ["LLM_MODEL_ID"] = self.original_model_id
        elif "LLM_MODEL_ID" in os.environ:
            del os.environ["LLM_MODEL_ID"]

        if self.original_api_key is not None:
            os.environ["LLM_API_KEY"] = self.original_api_key
        elif "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]

        if self.original_base_url is not None:
            os.environ["LLM_BASE_URL"] = self.original_base_url
        elif "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]

        if self.original_timeout is not None:
            os.environ["LLM_TIMEOUT"] = self.original_timeout
        elif "LLM_TIMEOUT" in os.environ:
            del os.environ["LLM_TIMEOUT"]

    @patch('llm_client.OpenAI')
    def test_init_initializes_with_env_vars(self, mock_openai_class):
        """测试使用环境变量初始化"""
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance

        llm = HelloAgentsLLM()

        # 验证属性设置正确
        assert llm.model == "test-model"

        # 验证OpenAI客户端被正确初始化
        mock_openai_class.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://test.base.url",
            timeout=60  # 默认超时时间
        )

    @patch('llm_client.OpenAI')
    def test_init_initializes_with_passed_params(self, mock_openai_class):
        """测试使用传入参数初始化"""
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance

        llm = HelloAgentsLLM(
            model="custom-model",
            apiKey="custom-api-key",
            baseUrl="https://custom.base.url",
            timeout=120
        )

        # 验证属性设置为传入的参数
        assert llm.model == "custom-model"

        # 验证OpenAI客户端使用传入参数初始化
        mock_openai_class.assert_called_once_with(
            api_key="custom-api-key",
            base_url="https://custom.base.url",
            timeout=120
        )

    @patch('llm_client.OpenAI')
    def test_init_uses_env_vars_when_param_is_none(self, mock_openai_class):
        """测试参数为None时使用环境变量"""
        mock_client_instance = Mock()
        mock_openai_class.return_value = mock_client_instance

        llm = HelloAgentsLLM(model=None, apiKey=None, baseUrl=None)

        # 验证属性仍然使用环境变量
        assert llm.model == "test-model"

        # 验证OpenAI客户端使用环境变量参数初始化
        mock_openai_class.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://test.base.url",
            timeout=60
        )

    @patch('llm_client.OpenAI')
    def test_init_raises_error_when_required_params_missing(self, mock_openai_class):
        """测试必需参数缺失时抛出错误"""
        # 删除环境变量以触发错误
        if "LLM_MODEL_ID" in os.environ:
            del os.environ["LLM_MODEL_ID"]
        if "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]
        if "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]

        with pytest.raises(ValueError) as exc_info:
            HelloAgentsLLM()

        assert "模型ID、API密钥和服务地址必须被提供或在.env文件中定义。" in str(exc_info.value)

    @patch('llm_client.OpenAI')
    def test_think_calls_chat_completions_create(self, mock_openai_class):
        """测试think方法调用chat.completions.create"""
        mock_client_instance = Mock()
        mock_chat_instance = Mock()
        mock_response = Mock()

        # 设置模拟链式调用
        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat = mock_chat_instance
        mock_chat_instance.completions = Mock()
        mock_chat_instance.completions.create.return_value = iter([
            Mock(choices=[Mock(delta=Mock(content="Hello "))]),
            Mock(choices=[Mock(delta=Mock(content="world!"))]),
            Mock(choices=[Mock(delta=Mock(content=""))])
        ])

        llm = HelloAgentsLLM()

        messages = [{"role": "user", "content": "测试消息"}]
        result = llm.think(messages=messages)

        # 验证chat.completions.create被正确调用
        mock_chat_instance.completions.create.assert_called_once_with(
            model="test-model",
            messages=messages,
            temperature=0,
            stream=True
        )

    @patch('llm_client.OpenAI')
    def test_think_handles_streaming_response(self, mock_openai_class):
        """测试处理流式响应"""
        mock_client_instance = Mock()
        mock_chat_instance = Mock()

        # 创建模拟的流式响应
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="第一部分 "))]),
            Mock(choices=[Mock(delta=Mock(content="第二部分 "))]),
            Mock(choices=[Mock(delta=Mock(content="第三部分"))]),
            Mock(choices=[Mock(delta=Mock(content=""))])  # 结束
        ]

        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat = mock_chat_instance
        mock_chat_instance.completions = Mock()
        mock_chat_instance.completions.create.return_value = iter(mock_chunks)

        llm = HelloAgentsLLM()

        messages = [{"role": "user", "content": "测试消息"}]
        result = llm.think(messages=messages)

        # 验证响应被正确拼接
        assert result == "第一部分 第二部分 第三部分"

    @patch('llm_client.OpenAI')
    def test_think_returns_none_on_exception(self, mock_openai_class):
        """测试异常情况下返回None"""
        mock_client_instance = Mock()
        mock_chat_instance = Mock()

        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat = mock_chat_instance
        mock_chat_instance.completions = Mock()
        mock_chat_instance.completions.create.side_effect = Exception("API错误")

        llm = HelloAgentsLLM()

        messages = [{"role": "user", "content": "测试消息"}]
        result = llm.think(messages=messages)

        # 验证异常情况下返回None
        assert result is None

    @patch('llm_client.OpenAI')
    def test_think_uses_custom_temperature(self, mock_openai_class):
        """测试使用自定义温度参数"""
        mock_client_instance = Mock()
        mock_chat_instance = Mock()

        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="响应"))]),
            Mock(choices=[Mock(delta=Mock(content=""))])
        ]

        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat = mock_chat_instance
        mock_chat_instance.completions = Mock()
        mock_chat_instance.completions.create.return_value = iter(mock_chunks)

        llm = HelloAgentsLLM()

        messages = [{"role": "user", "content": "测试消息"}]
        result = llm.think(messages=messages, temperature=0.7)

        # 验证temperature参数被正确传递
        mock_chat_instance.completions.create.assert_called_once_with(
            model="test-model",
            messages=messages,
            temperature=0.7,
            stream=True
        )

    @patch('llm_client.OpenAI')
    def test_think_handles_empty_delta_content(self, mock_openai_class):
        """测试处理空的delta内容"""
        mock_client_instance = Mock()
        mock_chat_instance = Mock()

        # 模拟包含空内容的响应
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="部分内容"))]),
            Mock(choices=[Mock(delta=Mock(content=""))]),  # 空内容
            Mock(choices=[Mock(delta=Mock(content="更多内容"))]),
            Mock(choices=[Mock(delta=Mock(content=""))])
        ]

        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat = mock_chat_instance
        mock_chat_instance.completions = Mock()
        mock_chat_instance.completions.create.return_value = iter(mock_chunks)

        llm = HelloAgentsLLM()

        messages = [{"role": "user", "content": "测试消息"}]
        result = llm.think(messages=messages)

        # 验证空内容被跳过，只有有效内容被拼接
        assert result == "部分内容更多内容"


class TestHelloAgentsLLMIntegration:
    """HelloAgentsLLM集成测试"""

    def setup_method(self):
        """测试方法执行前的设置"""
        # 保存原始环境变量
        self.original_model_id = os.environ.get("LLM_MODEL_ID")
        self.original_api_key = os.environ.get("LLM_API_KEY")
        self.original_base_url = os.environ.get("LLM_BASE_URL")

        # 设置测试用的环境变量
        os.environ["LLM_MODEL_ID"] = "test-model-for-integration"
        os.environ["LLM_API_KEY"] = "test-api-key-for-integration"
        os.environ["LLM_BASE_URL"] = "https://test-integration.base.url"

    def teardown_method(self):
        """测试方法执行后的清理"""
        # 恢复原始环境变量
        if self.original_model_id is not None:
            os.environ["LLM_MODEL_ID"] = self.original_model_id
        elif "LLM_MODEL_ID" in os.environ:
            del os.environ["LLM_MODEL_ID"]

        if self.original_api_key is not None:
            os.environ["LLM_API_KEY"] = self.original_api_key
        elif "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]

        if self.original_base_url is not None:
            os.environ["LLM_BASE_URL"] = self.original_base_url
        elif "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]

    @patch('llm_client.OpenAI')
    def test_full_initialization_and_think_flow(self, mock_openai_class):
        """测试完整的初始化和think流程"""
        # 设置模拟响应
        mock_client_instance = Mock()
        mock_chat_instance = Mock()

        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="完整 "))]),
            Mock(choices=[Mock(delta=Mock(content="流程 "))]),
            Mock(choices=[Mock(delta=Mock(content="测试"))]),
            Mock(choices=[Mock(delta=Mock(content=""))])
        ]

        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat = mock_chat_instance
        mock_chat_instance.completions = Mock()
        mock_chat_instance.completions.create.return_value = iter(mock_chunks)

        # 完整的初始化和调用流程
        llm = HelloAgentsLLM()

        assert llm.model == "test-model-for-integration"
        assert llm.client is not None

        messages = [{"role": "user", "content": "完整流程测试"}]
        result = llm.think(messages=messages)

        assert result == "完整 流程 测试"


if __name__ == '__main__':
    pytest.main([__file__])