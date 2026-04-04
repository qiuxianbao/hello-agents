import numpy as np

"""
np（Numerical Python）是 Python 科学计算的核心基础库
运行设备：CPU
"""


def test_np():
    # 一维数组（向量）
    print(np.array([0.9, 0.8]))

    # 二维数组(矩阵)
    print(np.array([[1, 2], [3, 4]]))

    # 三维数组 (张量)
    print(np.array([[[1, 2]], [[3, 4]]]))

    # 计算平方和
    vec1 = np.array([0.9, 0.2])
    print(np.sum(vec1 ** 2))  # ** 是幂运算符，相当于 pow()


def _main():
    test_np()


if __name__ == '__main__':
    _main()
