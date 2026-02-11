from hello_agents.tools import MCPTool, A2ATool, ANPTool

# 1. MCP：访问工具
def test_mcp():
    mcp_tool = MCPTool()
    result = mcp_tool.run({
        "action": "call_tool",
        "tool_name": "add",
        "arguments": {"a": 10, "b": 20}
    })
    print(f"MCP计算结果: {result}")  # 输出: 30.0


# 2. A2A：智能体通信
def test_a2a():
    a2a_tool = A2ATool("http://localhost:5000")
    print("A2A工具创建成功")

# 3. ANP：服务发现
def test_anp():
    anp_tool = ANPTool()
    anp_tool.run({
        "action": "register_service",
        "service_id": "calculator",
        "service_type": "math",
        "endpoint": "http://localhost:8080"
    })
    services = anp_tool.run({"action": "discover_services"})
    print(f"发现的服务: {services}")

if __name__ == "__main__":
    test_mcp()
    # test_a2a()
    # test_anp()
