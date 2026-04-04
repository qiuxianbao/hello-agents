import collections
from typing import Iterator


def test_n_gram():
    """
    N-gram 语言模型示例 - 使用 Bigram (N=2) 模型计算句子概率

    示例语料库："datawhale agent learns datawhale agent works"
    目标：估算句子 "datawhale agent learns" 出现的概率

    计算步骤：
    第一步：计算第一个词的概率 P(datawhale)
      - datawhale 出现了 2 次，总词数是 6
      - P(datawhale) = 2/6 ≈ 0.333

    第二步：计算条件概率 P(agent | datawhale)
      - 词对 "datawhale agent" 出现了 2 次
      - datawhale 出现了 2 次
      - P(agent|datawhale) = 2/2 = 1

    第三步：计算条件概率 P(learns | agent)
      - 词对 "agent learns" 出现了 1 次
      - agent 出现了 2 次
      - P(learns|agent) = 1/2 = 0.5

    最后：将概率连乘，整个句子的近似概率为：
      - P(datawhale agent learns) ≈ P(datawhale) · P(agent|datawhale) · P(learns|agent)
      - ≈ 0.333 · 1 · 0.5 ≈ 0.167
    """

    # 示例语料库，与上方案例讲解中的语料库保持一致
    corpus = "datawhale agent learns datawhale agent works"
    tokens = corpus.split()
    total_tokens = len(tokens)

    # --- 第一步：计算 P(datawhale) ---
    # 知识点：查看List中元素出现的次数
    count_datawhale = tokens.count('datawhale')
    p_datawhale = count_datawhale / total_tokens
    print(f"第一步: P(datawhale) = {count_datawhale}/{total_tokens} = {p_datawhale:.3f}")

    # --- 第二步：计算 P(agent|datawhale) ---
    # 先计算 bigrams 用于后续步骤
    # 知识点：zip 交叉构建一个元组，[('datawhale', 'agent'), ('agent', 'learns'), ('learns', 'datawhale'), ('datawhale', 'agent'), ('agent', 'works')]
    bigrams = zip(tokens, tokens[1:])
    # print(f"第二步: bigrams = {list(bigrams)}")      # 此处不能打印日志，因为zip返回值是一个迭代器对象

    #  Counter({('datawhale', 'agent'): 2, ('agent', 'learns'): 1, ('learns', 'datawhale'): 1, ('agent', 'works'): 1})
    bigram_counts = collections.Counter(bigrams)
    print(f"第二步: bigram_counts = {bigram_counts}")

    count_datawhale_agent = bigram_counts[('datawhale', 'agent')]
    # count_datawhale 已在第一步计算
    p_agent_given_datawhale = count_datawhale_agent / count_datawhale
    print(f"第二步: P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.3f}")

    # --- 第三步：计算 P(learns|agent) ---
    count_agent_learns = bigram_counts[('agent', 'learns')]
    count_agent = tokens.count('agent')
    p_learns_given_agent = count_agent_learns / count_agent
    print(f"第三步: P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.3f}")

    # --- 最后：将概率连乘 ---
    p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
    print(
        f"最后: P('datawhale agent learns') ≈ {p_datawhale:.3f} * {p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = {p_sentence:.3f}")


def test_zip():
    """
    Yield tuples until an input is exhausted

    zip() 会以最短的 iterable 为准进行打包

    'abcdefg' - 一个包含 7 个字符的字符串
    range(3) - 生成序列 [0, 1, 2]（3 个元素）
    range(4) - 生成序列 [0, 1, 2, 3]（4 个元素）

    :return:
    """
    z = zip('abcdefg', range(3), range(4))  # 返回的是一个迭代器
    print(list(z))
    print(list(z))  # []，因为迭代器已经用完，所以返回空列表

    #
    z = zip('abcdefg', range(3), range(4))
    print(type(z))
    print(isinstance(z, Iterator))  # True

    l = list(z)  # 保存为list，可以打印多次
    print(l)
    print(l)


def _main():
    test_n_gram()
    # test_zip()


if __name__ == '__main__':
    _main()
