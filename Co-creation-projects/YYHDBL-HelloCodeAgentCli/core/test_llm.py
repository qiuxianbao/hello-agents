import pytest


class TestHelloAgentsLLM:

    """测试生成器"""
    def sub_gen(self):
        yield 1
        yield 2

    def main_gen(self):
        # 知识点：yield是定义生成器函数的关键字，执行到yield语句时，会暂停并返回值，再次调用时从暂停处继续执行
        # 等价于 for i in self.sub_gen(): yield i
        yield from self.sub_gen()


    def test_sub_gen(self):
        print(list(self.main_gen()))


if __name__ == "__main__":
    pytest.main([__file__])
