import numpy as np


class TestNp:


    def test_(self):
        pass



    def test_l2(self):
        import numpy as np
        # 定义一个向量
        vector = np.array([0.423, 0.525, 0.525])
        # 计算L2范数（向量中每个元素的平方和的平方根）
        l2_norm = np.sqrt(np.sum(vector ** 2))
        print(l2_norm)

        # 进行L2归一化（每个向量都被转换为在单位球面上的向量，即它们的长度都被缩放到1）
        normalized_vector = vector / l2_norm
        print(normalized_vector)


    def test_log(self):
        # np.log == ln, 实际上是np.log10(5/2)
        print(np.log(5 / 2))
