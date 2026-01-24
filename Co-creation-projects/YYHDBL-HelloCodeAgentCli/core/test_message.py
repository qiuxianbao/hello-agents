from typing import Literal

import pytest
from pydantic import BaseModel
from message import Message

# TODO：
class TestLiteral:
    def set_color(self, color: Literal["red", "green", "blue"]):
        print(f"Selected color:{color}")


"""
知识点：BaseModel 是python提供的核心基类，用于数据校验、序列化和反序列化  
"""
class User(BaseModel):
    name: str
    age: int
    email: str | None = None


class TestUser:

    def test_user(self):
        user = User(name="alice", age=25)
        # name='Bob' age=25 email=None
        print(user)

        # 转Json
        # {"name":"alice","age":25,"email":null}
        print(user.model_dump_json())

        # 转字典
        # {'name': 'alice', 'age': 25, 'email': None}
        user_dict = user.model_dump()
        print(user_dict)

        # 字典转对象
        user2 = User(**user_dict)
        print(user2)


class TestMyMessage:
    def test_to_dict(self):
        msg = Message("who are you", "user")
        """
        TODO：导入HelloAgent#Message没问题，当前包就有问题
        
        test_message.py:None (test_message.py)
        ImportError while importing test module 'C:\VsCode\llm\hello-agents\Co-creation-projects\YYHDBL-HelloCodeAgentCli\core\test_message.py'.
        Hint: make sure your test modules/packages have valid Python names.
        Traceback:
        C:\Python\Python311\Lib\site-packages\_pytest\python.py:507: in importtestmodule
        """
        print(msg.to_dict())

if __name__ == "__main__":
    pytest.main([__file__])
