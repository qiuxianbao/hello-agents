# import pytest
# import re
# from unittest.mock import Mock, patch, MagicMock
# from ReAct import ReActAgent, REACT_PROMPT_TEMPLATE
# from tools import ToolExecutor
# from llm_client import HelloAgentsLLM
#
#
# class TestReActAgent:
#     """ReActAgent类的单元测试"""
#
#     def setup_method(self):
#         """测试方法执行前的设置"""
#         # 创建Mock对象
#         self.mock_llm_client = Mock(spec=HelloAgentsLLM)
#         self.mock_tool_executor = Mock(spec=ToolExecutor)
#
#         # 设置默认返回值
#         self.mock_llm_client.think.return_value = "Thought: 测试思考\nAction: Search[test query]"
#         self.mock_tool_executor.getAvailableTools.return_value = "- Search: 搜索工具"
#         self.mock_tool_executor.getTool.return_value = lambda x: f"搜索结果: {x}"
#
#     def test_init_initializes_correctly(self):
#         """测试ReActAgent初始化"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor,
#             max_steps=5
#         )
#
#         assert agent.llm_client == self.mock_llm_client
#         assert agent.tool_executor == self.mock_tool_executor
#         assert agent.max_steps == 5
#         assert agent.history == []
#
#     def test_run_successful_completion(self):
#         """测试成功完成的运行流程"""
#         # 设置模拟的LLM响应序列
#         responses = [
#             "Thought: 我需要搜索华为最新手机\nAction: Search[华为最新手机]",
#             "Thought: 我找到了一些信息\nAction: Finish[华为最新手机是Mate 60系列，主要卖点是麒麟芯片回归]",
#         ]
#
#         self.mock_llm_client.think.side_effect = responses
#         self.mock_tool_executor.getTool.return_value = lambda x: f"搜索结果: {x}"
#
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor,
#             max_steps=5
#         )
#
#         result = agent.run("华为最新的手机是什么？")
#
#         assert result == "华为最新手机是Mate 60系列，主要卖点是麒麟芯片回归"
#         assert self.mock_llm_client.think.call_count == 2  # 被调用2次
#
#     def test_run_exceeds_max_steps(self):
#         """测试超过最大步骤数的情况"""
#         # 设置持续返回非完成指令
#         self.mock_llm_client.think.return_value = "Thought: 继续搜索\nAction: Search[继续查询]"
#         self.mock_tool_executor.getTool.return_value = lambda x: f"搜索结果: {x}"
#
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor,
#             max_steps=2
#         )
#
#         result = agent.run("测试问题")
#
#         assert result is None  # 达到最大步骤数时返回None
#         assert self.mock_llm_client.think.call_count == 2  # 被调用2次（等于max_steps）
#
#     def test_run_handles_invalid_action(self):
#         """测试处理无效行动的情况"""
#         invalid_response = "Thought: 我想执行一个不存在的行动\nAction: InvalidTool[invalid input]"
#         self.mock_llm_client.think.return_value = invalid_response
#         self.mock_tool_executor.getTool.return_value = None  # 工具不存在
#
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor,
#             max_steps=3
#         )
#
#         result = agent.run("测试问题")
#
#         # 应该继续尝试直到达到最大步骤数
#         assert result is None
#
#     def test_run_handles_empty_llm_response(self):
#         """测试处理空LLM响应的情况"""
#         self.mock_llm_client.think.return_value = ""
#
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor,
#             max_steps=3
#         )
#
#         result = agent.run("测试问题")
#
#         assert result is None
#
#     def test_run_handles_no_action_found(self):
#         """测试无法解析行动的情况"""
#         invalid_response = "这是一个无效的响应格式，没有正确的Action"
#         self.mock_llm_client.think.return_value = invalid_response
#
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor,
#             max_steps=3
#         )
#
#         result = agent.run("测试问题")
#
#         assert result is None
#
#     def test_parse_output_correctly_parses_thought_and_action(self):
#         """测试正确解析思考和行动"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         text = "Thought: 这是思考过程\nAction: Search[测试查询]"
#         thought, action = agent._parse_output(text)
#
#         assert thought == "这是思考过程"
#         assert action == "Search[测试查询]"
#
#     def test_parse_output_handles_missing_thought(self):
#         """测试处理缺失思考的情况"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         text = "Action: Search[测试查询]"
#         thought, action = agent._parse_output(text)
#
#         assert thought is None
#         assert action == "Search[测试查询]"
#
#     def test_parse_output_handles_missing_action(self):
#         """测试处理缺失行动的情况"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         text = "Thought: 这是思考过程"
#         thought, action = agent._parse_output(text)
#
#         assert thought == "这是思考过程"
#         assert action is None
#
#     def test_parse_action_correctly_parses_valid_action(self):
#         """测试正确解析有效的行动"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         action_text = "Search[测试输入]"
#         tool_name, tool_input = agent._parse_action(action_text)
#
#         assert tool_name == "Search"
#         assert tool_input == "测试输入"
#
#     def test_parse_action_handles_invalid_action_format(self):
#         """测试处理无效行动格式的情况"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         action_text = "InvalidFormat"
#         tool_name, tool_input = agent._parse_action(action_text)
#
#         assert tool_name is None
#         assert tool_input is None
#
#     def test_parse_action_input_correctly_parses_input(self):
#         """测试正确解析行动输入"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         action_text = "Finish[最终答案]"
#         input_result = agent._parse_action_input(action_text)
#
#         assert input_result == "最终答案"
#
#     def test_parse_action_input_handles_complex_input(self):
#         """测试处理复杂输入的情况"""
#         agent = ReActAgent(
#             llm_client=self.mock_llm_client,
#             tool_executor=self.mock_tool_executor
#         )
#
#         action_text = "Search[复杂的输入: 包含特殊字符!@#$%]"
#         input_result = agent._parse_action_input(action_text)
#
#         assert input_result == "复杂的输入: 包含特殊字符!@#$%"
#
#
# class TestReActPromptTemplate:
#     """REACT_PROMPT_TEMPLATE的测试"""
#
#     def test_prompt_template_contains_required_placeholders(self):
#         """测试提示模板包含必需的占位符"""
#         template = REACT_PROMPT_TEMPLATE
#
#         assert "{tools}" in template
#         assert "{question}" in template
#         assert "{history}" in template
#
#         # 检查格式说明是否存在
#         assert "Thought:" in template
#         assert "Action:" in template
#         assert "{{tool_name}}[{{tool_input}}]" in template
#         assert "Finish[最终答案]" in template
#
#
# class TestToolExecutor:
#     """ToolExecutor类的单元测试"""
#
#     def test_register_tool_adds_tool_correctly(self):
#         """测试正确添加工具"""
#         executor = ToolExecutor()
#         mock_func = Mock()
#
#         executor.registerTool("TestTool", "测试工具描述", mock_func)
#
#         assert "TestTool" in executor.tools
#         assert executor.tools["TestTool"]["description"] == "测试工具描述"
#         assert executor.tools["TestTool"]["func"] == mock_func
#
#     def test_get_tool_returns_correct_function(self):
#         """测试正确获取工具函数"""
#         executor = ToolExecutor()
#         mock_func = Mock(return_value="执行结果")
#         executor.registerTool("TestTool", "测试工具描述", mock_func)
#
#         retrieved_func = executor.getTool("TestTool")
#
#         assert retrieved_func == mock_func
#
#         # 测试执行函数
#         result = retrieved_func("测试输入")
#         assert result == "执行结果"
#
#     def test_get_tool_returns_none_for_nonexistent_tool(self):
#         """测试获取不存在工具时返回None"""
#         executor = ToolExecutor()
#
#         result = executor.getTool("NonExistentTool")
#
#         assert result is None
#
#     def test_get_available_tools_formats_correctly(self):
#         """测试正确格式化可用工具"""
#         executor = ToolExecutor()
#         mock_func = Mock()
#         executor.registerTool("TestTool", "测试工具描述", mock_func)
#         executor.registerTool("AnotherTool", "另一个工具描述", mock_func)
#
#         result = executor.getAvailableTools()
#
#         assert "- TestTool: 测试工具描述" in result
#         assert "- AnotherTool: 另一个工具描述" in result
#
#
# # Mock测试 - 测试完整的工作流程
# @patch('ReAct.HelloAgentsLLM')
# @patch('ReAct.ToolExecutor')
# def test_end_to_end_flow_with_mock(mock_tool_executor_class, mock_llm_client_class):
#     """端到端流程测试（使用Mock）"""
#     # 设置Mock实例
#     mock_llm_instance = Mock()
#     mock_tool_executor_instance = Mock()
#
#     mock_llm_client_class.return_value = mock_llm_instance
#     mock_tool_executor_class.return_value = mock_tool_executor_instance
#
#     # 设置Mock行为
#     responses = [
#         "Thought: 我需要搜索信息\nAction: Search[测试查询]",
#         "Thought: 我得到了结果\nAction: Finish[最终答案]"
#     ]
#     mock_llm_instance.think.side_effect = responses
#
#     mock_tool_executor_instance.getAvailableTools.return_value = "- Search: 搜索工具"
#     mock_tool_executor_instance.getTool.return_value = lambda x: f"搜索结果: {x}"
#
#     # 创建agent并运行
#     agent = ReActAgent(
#         llm_client=mock_llm_instance,
#         tool_executor=mock_tool_executor_instance,
#         max_steps=5
#     )
#
#     result = agent.run("测试问题")
#
#     assert result == "最终答案"
#     assert mock_llm_instance.think.call_count == 2
#
#
# if __name__ == '__main__':
#     pytest.main([__file__])