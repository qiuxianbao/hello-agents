"""工具基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

# 为了支持复杂的参数验证和文档生成
class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

# Abstract Base Classes
class Tool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.dict() for param in self.get_parameters()]
        }

    # 知识点：print(obj)优先调用__str__
    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    # 知识点：representation
    # 交互式查看优先调用__repr__
    def __repr__(self) -> str:
        return self.__str__()