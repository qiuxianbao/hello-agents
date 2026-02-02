import pytest

class TestMyLLM:

    # 知识点：可变参数，1个*
    # 会将传入的多个关键字参数打包成一个元组，# <class 'tuple'>
    # tuple和list区别：list是可变对象，tuple是只读对象
    def args_tuple(self, *args):
        pass    #占位符，表示什么都不做，用来避免语法错误
        print(type(args))
        print(list(args))

    def test_args_tuple(self):
        self.args_tuple(1, [2, 3], "a")


    # 知识点：可变参数，2个**
    # 会将传入的多个关键字参数打包成一个字典，# <class 'dict'>
    def kwargs_dict(self, **kwargs):
        print(type(kwargs))
        print(kwargs)

        kwargs.update({"d": 4})
        print(kwargs)

        kwargs.update({"d": 5})
        print(kwargs)

    def test_dict_kwargs(self):
        self.kwargs_dict(a=1, b=2, c=3)

if __name__ == "__main__":
    pytest.main([__file__])
