import asyncio
import json
import os
import sys

try:
    from .server import MCPServer
    from .client import MCPClient
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from server import MCPServer
    from client import MCPClient


class TestClient:


    def test_connect_http(self):

        async def _run():
            """
            对应 server.py #server.run(transport="http", host="127.0.0.1", port=8081)
            :return:
            """
            client = MCPClient("http://127.0.0.1:8081/mcp")
            async with client:
                tools = await client.list_tools()
                return tools

        tools = asyncio.run(_run())

        # 📋 Found 2 tool(s):
        print(f"📋 Found {len(tools)} tool(s):")
        print(tools)


    def test_connect_memory(self):
        """测试 client 连接 server 并列举所有 tools"""

        # 1. 创建 server 并注册工具
        server = MCPServer(name="test-server", description="Test MCP Server")

        def add(a: int, b: int) -> str:
            """两数相加"""
            return str(a + b)

        server.add_tool(add, name="add", description="Add two numbers")

        # 2. 用内存传输创建 client，连接 server 并列举 tools
        async def _run():
            # 直接传入 FastMCP 实例，走内存传输

            """
            创建实例
            🧠 使用内存传输: test - server
            :return:
            """
            client = MCPClient(server.mcp)

            """
            知识点：aenter/aexit
            aenter 是Python异步上下文管理器协议的一部分，当使用 async with 语句时自动调用。
                
            🔗 连接到 MCP服务器
            ✅ 连接成功！
            🔌 连接已断开
            """
            # 当执行这行代码时，__aenter__ 会被自动调用
            async with client:
                tools = await client.list_tools()
                return tools
            # 当离开这个代码块时，__aexit__ 会被调用

        tools = asyncio.run(_run())

        # 3. 验证
        print(f"\n🔧 Server: {server.name}")
        # 📋 Found 1 tool(s):
        print(f"📋 Found {len(tools)} tool(s):")

        """
        [
            {
                'name': 'add',
                'description': 'Add two numbers',
                'input_schema': {
                    'properties': {
                        'a': {
                            'type': 'integer'
                        },
                        'b': {
                            'type': 'integer'
                        }
                    },
                    'required': [
                        'a',
                        'b'
                    ],
                    'type': 'object'
                }
            }
        ]
        """
        print(tools)
