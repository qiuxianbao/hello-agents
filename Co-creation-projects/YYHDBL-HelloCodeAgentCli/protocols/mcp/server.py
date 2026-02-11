"""
基于 fastmcp 库的 MCP 服务器实现

使用 fastmcp 库实现 Model Context Protocol 服务器功能。
fastmcp 是一个快速创建 MCP 服务器的 Python 库。
"""

from typing import Dict, Any, List, Optional, Callable
try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "fastmcp is required for MCP server functionality. "
        "Install it with: pip install fastmcp"
    )


class MCPServer:
    """基于 fastmcp 库的 MCP 服务器"""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None
    ):
        """
        初始化 MCP 服务器
        
        Args:
            name: 服务器名称
            description: 服务器描述
        """
        # 知识点：MCP服务器
        self.mcp = FastMCP(name=name)
        self.name = name
        self.description = description or f"{name} MCP Server"
        
    def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加工具到服务器
        
        Args:
            func: 工具函数
            name: 工具名称（可选，默认使用函数名）
            description: 工具描述（可选，默认使用函数文档字符串）
        """
        # 使用装饰器注册工具
        if name or description:
            """
            示例：
            >>> self.mcp.tool() > Callable[[AnyFunction], FunctionTool] | FunctionTool
            这是一个可调用对象（callable） 的类型注解。
            它表示一个函数或类，接受一个参数（类型为 AnyFunction），并返回一个 FunctionTool 类型的对象
            
            原理：
            最终会添加到ToolManager#self._tools: dict[str, Tool] = {} 
            """
            self.mcp.tool(name=name, description=description)(func)
        else:
            self.mcp.tool()(func)
        
    def add_resource(
        self,
        func: Callable,
        uri: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加资源到服务器
        
        Args:
            func: 资源处理函数
            uri: 资源 URI（可选）
            name: 资源名称（可选）
            description: 资源描述（可选）
        """
        # 使用装饰器注册资源
        if uri:
            self.mcp.resource(uri)(func)
        else:
            self.mcp.resource()(func)
        
    def add_prompt(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加提示词模板到服务器
        
        Args:
            func: 提示词生成函数
            name: 提示词名称（可选）
            description: 提示词描述（可选）
        """
        # 使用装饰器注册提示词
        if name or description:
            self.mcp.prompt(name=name, description=description)(func)
        else:
            self.mcp.prompt()(func)
        
    def run(self, transport: str = "stdio", **kwargs):
        """运行服务器

        Args:
            transport: 传输方式 ("stdio", "http", "sse")
            **kwargs: 传输特定的参数
                - host: HTTP 服务器主机（默认 "127.0.0.1"）
                - port: HTTP 服务器端口（默认 8000）
                - 其他 FastMCP.run() 支持的参数

        Examples:
            # Stdio 传输（默认）
            server.run()

            # HTTP 传输
            server.run(transport="http", host="0.0.0.0", port=8081)

            # SSE 传输
            server.run(transport="sse", host="0.0.0.0", port=8081)

        原理：
        启动一个基于 Model Context Protocol (MCP) 的服务器，根据指定传输方式处理，进行监听
            >>> fastmcp.server.server.FastMCP.run_async
        1.stdio
        使用 stdio_server() 创建读写流（read_stream 和 write_stream）。
        调用底层 self._mcp_server.run() 方法，传入读写流和初始化选项，开始监听客户端请求
            >>> fastmcp.server.server.FastMCP.run_stdio_async

        2.http/sse/streamable-http
        调用 http_app() 方法创建一个 Starlette 应用实例(不同的transport实例不同)
            >>> fastmcp.server.server.FastMCP.http_app
        基于 Starlette 和 Uvicorn 构建，启动 Uvicorn 服务器
            >>> config = uvicorn.Config(app, host=host, port=port, **config_kwargs)
            >>> server = uvicorn.Server(config)
            >>> server.serve()

        知识点：Uvicorn
        Uvicorn 是一个基于 uvloop 和 httptools 构建的超高速 ASGI（Asynchronous Server Gateway Interface）服务器，专为 Python 异步 Web 应用设计。
        它支持 HTTP/1.1、WebSocket，并计划支持 HTTP/2，可与 FastAPI、Starlette、Django Channels 等框架无缝配合。

        """
        self.mcp.run(transport=transport, **kwargs)
        
    def get_info(self) -> Dict[str, Any]:
        """
        获取服务器信息
        
        Returns:
            服务器信息字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "MCP"
        }


# 便捷的服务器构建器（链式调用）
class MCPServerBuilder:
    """MCP 服务器构建器，提供链式 API"""

    def __init__(self, name: str, description: Optional[str] = None):
        self.server = MCPServer(name, description)

    # 知识点：返回值需要加 单引号
    def with_tool(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加工具（链式调用）"""
        self.server.add_tool(func, name, description)
        return self
        
    def with_resource(self, func: Callable, uri: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加资源（链式调用）"""
        self.server.add_resource(func, uri, name, description)
        return self
        
    def with_prompt(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加提示词（链式调用）"""
        self.server.add_prompt(func, name, description)
        return self
        
    def build(self) -> MCPServer:
        """构建服务器"""
        return self.server
        
    def run(self):
        """构建并运行服务器"""
        self.server.run()


# 示例：创建一个简单的 MCP 服务器
def create_example_server() -> MCPServer:
    """创建一个示例 MCP 服务器"""
    server = MCPServer(
        name="example-server",
        description="A simple example MCP server with calculator and greeting tools"
    )
    
    # 添加一个简单的计算器工具
    def calculator(expression: str) -> str:
        """计算数学表达式
        
        Args:
            expression: 要计算的数学表达式，例如 "2 + 2" 或 "10 * 5"
        """
        try:
            # 安全的表达式求值（仅支持基本运算）
            allowed_chars = set("0123456789+-*/() .")
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    server.add_tool(calculator, name="calculator", description="Calculate a mathematical expression")
    
    # 添加一个问候工具
    def greet(name: str) -> str:
        """生成友好的问候语
        
        Args:
            name: 要问候的人的名字
        """
        return f"Hello, {name}! Welcome to the MCP server example."
    
    server.add_tool(greet, name="greet", description="Generate a friendly greeting")
    
    return server


if __name__ == "__main__":
    # 创建并运行示例服务器
    server = create_example_server()
    print(f"🚀 Starting {server.name}...")
    print(f"📝 {server.description}")
    print(f"🔌 Protocol: MCP")
    print(f"📡 Transport: stdio")
    print()
    # server.run()
    server.run(transport="http", host="127.0.0.1", port=8081)


