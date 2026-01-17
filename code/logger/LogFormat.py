import json
from typing import Any


class LogFormat:
    @staticmethod
    def format_print(data: Any, title: str = "Log Output", width: int = 50):
        """
        格式化打印数据
        :param data: 要打印的数据
        :param title: 标题
        :param width: 边框宽度
        """
        print(f"\n{'=' * width} {title} {'=' * width}")
        
        if isinstance(data, (dict, list)):
            # 如果是字典或列表，使用JSON美化输出
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(data)

        print('=' * ( 2 + width * 2 + len(title)))


if __name__ == '__main__':
    LogFormat.format_print("hello world")
    LogFormat.format_print("content", "Title")
    LogFormat.format_print({"name": "张三", "age": 18}, "JSON")