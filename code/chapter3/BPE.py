import re, collections

"""
主流的子词切分算法，用于进行tokennization（将输入文本拆分为token）

Byte Pair Encoding	基于字节对编码
通过迭代合并最高频的相邻词元对，来逐步构建和扩充词表

核心：贪心合并过程
1. 初始化：将词表初始化为所有在语料库中出现过的基本字符
2. 迭代合并：在语料库上，统计所有【相邻词元对】的出现频率，找到频率最高的一对，将它们合并成一个新的词元，并加入词表
3. 重复：重复第 2 步，直到词表大小达到预设的阈值。

"""


def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        # 对word进行拆分
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # 统计词对出现的次数
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """合并词元对"""
    v_out = {}
    bigram = re.escape(' '.join(pair))  # escape()方法对正则表达式中的特殊字符进行转义
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')  # 精确匹配完整单词，确保前面和后面都不是非空白字符
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def test_bpe():
    # 准备语料库，每个词末尾加上</w>表示结束，并切分好字符
    vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
    print(f"语料库: {list(vocab.keys())}")
    print("-" * 20)

    num_merges = 4  # 设置合并次数

    for i in range(num_merges):
        # 获取词对频率
        pairs = get_stats(vocab)
        print(pairs)

        if not pairs:
            break

        # 找出词频最高的词对
        best = max(pairs, key=pairs.get)
        # 合并
        vocab = merge_vocab(best, vocab)

        print(f"第{i + 1}次合并: {best} -> {''.join(best)}")
        print(f"新词表（部分）: {list(vocab.keys())}")
        print("-" * 20)


def test_reg_sub():
    pair = ('u', 'g')
    bigram = re.escape(' '.join(pair))  # "u g"

    # 找到独立的 token 对
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    print(p)

    # 处理单词
    word = "h u g </w>"

    """
    替换字符串
    sub(replacement, string)：正则表达式的替换方法
        第一个参数：替换成的字符串
        第二个参数：要处理的原始字符串
    """
    w_out = p.sub(''.join(pair), word)

    print(f"原始: {word}")  # h u g </w>
    print(f"替换后: {w_out}")  # h ug </w>


def test_reg_compile():
    bigram = "h e"

    """
    知识点：正则，断言语法
    +------------------+------------------------+--------------------------+
    | 语法             | 名称                   | 作用                     |
    +------------------+------------------------+--------------------------+
    | (?<name>...)     | 命名捕获组             | 给捕获组命名             |
    | (?<!...)         | 反向否定断言           | 前面**不**匹配某模式     |
    | (?<=...)         | 反向肯定断言           | 前面**必须**匹配某模式   |
    | (?!...)          | 正向否定断言           | 后面**不**匹配某模式     |
    | (?=...)          | 正向肯定断言           | 后面**必须**匹配某模式   |
    +------------------+------------------------+--------------------------+
    """
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    text1 = "h e l l o"
    text2 = "h el l o"

    text3 = "t h e"
    text4 = "th e"

    text5 = "h e"

    # span(s,e): 包含头不包含尾
    print(p.search(text1))  # <re.Match object; span=(0, 3), match='h e'>
    print(p.search(text2))  # None
    print(p.search(text3))  # <re.Match object; span=(2, 5), match='h e'>
    print(p.search(text4))  # None
    print(p.search(text5))  # <re.Match object; span=(0, 3), match='h e'>


def _main():
    # test_reg_compile()
    # test_reg_sub()
    test_bpe()


if __name__ == '__main__':
    _main()
