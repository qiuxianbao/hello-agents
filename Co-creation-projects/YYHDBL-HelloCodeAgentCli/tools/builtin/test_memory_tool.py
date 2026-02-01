import os
from typing import List


class TestMemoryTool:

    def test_memory_tool(self):
        from .memory_tool import MemoryTool

        # TODO: 没有调通
        memory_tool = MemoryTool()
        memory_tool.execute("add",
                            content="用户刚才问了关于Python函数的问题",
                            memory_type="working",
                            importance=0.6
                            )

        # 2. 情景记忆 - 具体事件和经历
        memory_tool.execute("add",
                            content="2024年3月15日，用户张三完成了第一个Python项目",
                            memory_type="episodic",
                            importance=0.8,
                            event_type="milestone",
                            location="在线学习平台"
                            )
        # 3. 语义记忆 - 抽象知识和概念
        memory_tool.execute("add",
                            content="Python是一种解释型、面向对象的编程语言",
                            memory_type="semantic",
                            importance=0.9,
                            knowledge_type="factual"
                            )
        # 4. 感知记忆 - 多模态信息
        memory_tool.execute("add",
                            content="用户上传了一张Python代码截图，包含函数定义",
                            memory_type="perceptual",
                            importance=0.7,
                            modality="image",
                            file_path="./uploads/code_screenshot.png"
                            )

    def test_os(self):
        db_dir = "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")


    def test_in_list(self):
        list = ["a", "b", "c"]
        print("a" in list)
        print("d" in list)


    def test_default(self):
        # 设置默认值
        perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]
        print(perceptual_memory_modalities)
