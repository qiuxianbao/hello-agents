from hello_agents.tools import RAGTool


class TestRag:

    def test_(self):
        pass


    def test_markitdown_convert(self):
        from markitdown import MarkItDown

        path = "C:/Users/admin/Desktop/Schedule.txt"
        markit_down = MarkItDown()
        text =  markit_down.convert(path)


    def test_execute(self):
        # TODO: 从文件中导入类报错
        #  from rag_tool import RAGTool
        #  E   ModuleNotFoundError: No module named 'rag_tool'
        rag_tool = RAGTool()
        result = rag_tool.execute("add_text",
                                  text="Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python的设计哲学强调代码的可读性和简洁的语法。",
                                  document_id="python_intro")
        print(f"知识: {result}")
