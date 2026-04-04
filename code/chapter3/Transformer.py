import math

import torch
import torch.nn as nn  # Neural Network（神经网络）


class MultiHeadAttention(nn.Module):
    """
    继承 nn.Module

    多头注意力机制模块： 允许模型同时关注不同位置的信息

    如果只进行一次上述的注意力计算（即单头），模型可能会只学会关注一种类型的关联。比如，在处理 "it" 时，可能只学会了关注主语。
    但语言中的关系是复杂的，我们希望模型能同时关注多种关系（如指代关系、时态关系、从属关系等）。

    多头注意力机制应运而生。它的思想很简单：把一次做完变成分成几组，分开做，再合并
    """

    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力机制

        参数:
            d_model (int): 词嵌入向量的维度，也是模型的整体维度
            num_heads (int): 注意力头的数量，用于并行计算不同子空间的注意力
        """
        super(MultiHeadAttention, self).__init__()
        # 确保维度可以均匀分割
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model  # 模型的总维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个注意力头的维度，// 含义是整除 比如：512//8=64

        # 定义 Q, K, V 和输出的线性变换层
        # 可学习的权重矩阵
        self.W_q = nn.Linear(d_model, d_model)  # 2个参数：in_features，out_features
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算

        参数:
            Q (Tensor): 查询 (Query, Q)：代表当前词元，它正在主动地“查询”其他词元以获取信息
            K (Tensor): 键 (Key, K)：代表句子中可被查询的词元“标签”或“索引”。
            V (Tensor): 值 (Value, V)：代表词元本身所携带的“内容”或“信息”。
            mask (Tensor, optional): 掩码矩阵，用于遮蔽不需要关注的位置，默认为 None

        返回:
            Tensor: 注意力输出，形状 (batch_size, num_heads, seq_len, d_k)
        """

        # 1. 计算注意力得分 (QK^T)
        """
        计算 查询向量（Query） 和 键向量（Key） 的点积
        
        知识点：
        transpose：转置
        K.transpose(-2, -1)：将矩阵的最后两个维度转置（行变列），这样才能和进行矩阵乘法
        除以每个头维度的平方根（即）√d_k 
        
        目的：缩放，是为了防止梯度消失
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 应用掩码，将需要遮蔽的位置设为负无穷
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重 (Softmax)
        # 稳定归一化
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加权求和 (权重 * V)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        将输入分割成多个注意力头

        参数:
            x (Tensor): 输入张量，形状 (batch_size, seq_length, d_model)

        返回:
            Tensor: 分割后的张量，形状 (batch_size, num_heads, seq_length, d_k)
        """
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        合并多个注意力头的输出

        参数:
            x (Tensor): 多头输出张量，形状 (batch_size, num_heads, seq_length, d_k)

        返回:
            Tensor: 合并后的张量，形状 (batch_size, seq_length, d_model)
        """
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        # contiguous 确保内存连续，避免视图错误
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        参数:
            Q (Tensor): Query 输入，形状 (batch_size, seq_len, d_model)
            K (Tensor): Key 输入，形状 (batch_size, seq_len, d_model)
            V (Tensor): Value 输入，形状 (batch_size, seq_len, d_model)
            mask (Tensor, optional): 注意力掩码，默认为 None

        返回:
            Tensor: 多头注意力输出，形状 (batch_size, seq_len, d_model)
        """
        # 1. 对 Q, K, V 进行线性变换
        # 通过权重矩阵生成 Q K V向量
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        # 计算相关得分
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化位置前馈网络

        参数:
            d_model (int): 输入和输出的维度
            d_ff (int): 前馈网络隐藏层的维度（通常比 d_model 大 4 倍）
            dropout (float): Dropout 比率，用于防止过拟合，默认 0.1
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 升维
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.linear2 = nn.Linear(d_ff, d_model)  # 降维
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x


class PositionalEncoding(nn.Module):
    """
    逐位置编码模块
    为输入序列的词嵌入向量添加位置编码。

    对于自注意力来说，“agent learns” 和 “learns agent” 这两个序列是完全等价的，因为它只关心词元之间的关系，而忽略了它们的排列。
    为了解决这个问题，Transformer 引入了位置编码 (Positional Encoding) 。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码

        参数:
            d_model (int): 词嵌入的维度
            dropout (float): Dropout 比率，默认 0.1
            max_len (int): 最大序列长度，位置编码矩阵的最大行数，默认 5000
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin
        # 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    编码器核心层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        初始化编码器层

        参数:
            d_model (int): 模型维度
            num_heads (int): 多头注意力的头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout 比率
        """
        super(EncoderLayer, self).__init__()
        # 多头注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        编码器层前向传播

        参数:
            x (Tensor): 输入张量，形状 (batch_size, seq_len, d_model)
            mask (Tensor): 源序列掩码，形状 (batch_size, 1, 1, src_len)

        返回:
            Tensor: 编码器层输出，形状 (batch_size, seq_len, d_model)
        """
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    解码器核心层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        初始化解码器层

        参数:
            d_model (int): 模型维度
            num_heads (int): 多头注意力的头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout 比率
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        解码器层前向传播

        参数:
            x (Tensor): 解码器输入，形状 (batch_size, tgt_seq_len, d_model)
            encoder_output (Tensor): 编码器输出，形状 (batch_size, src_seq_len, d_model)
            src_mask (Tensor): 源序列掩码，形状 (batch_size, 1, 1, src_len)
            tgt_mask (Tensor): 目标序列掩码，形状 (batch_size, 1, tgt_len, tgt_len)

        返回:
            Tensor: 解码器层输出，形状 (batch_size, tgt_seq_len, d_model)
        """
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len):
        """
        初始化编码器

        参数:
            vocab_size (int): 源语言词汇表大小
            d_model (int): 模型维度
            num_layers (int): 编码器层的数量
            num_heads (int): 多头注意力的头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout 比率
            max_len (int): 最大序列长度
        """
        super(Encoder, self).__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 编码层
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 残差连接，层化归一
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        编码器前向传播

        参数:
            x (Tensor): 输入 token IDs，形状 (batch_size, src_seq_len)
            mask (Tensor): 源序列掩码，形状 (batch_size, 1, 1, src_len)

        返回:
            Tensor: 编码器输出，形状 (batch_size, src_seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len):
        """
        初始化解码器

        参数:
            vocab_size (int): 目标语言词汇表大小
            d_model (int): 模型维度
            num_layers (int): 解码器层的数量
            num_heads (int): 多头注意力的头数
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout 比率
            max_len (int): 最大序列长度
        """
        super(Decoder, self).__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 解码层
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # 残差连接，层化归一
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        解码器前向传播

        参数:
            x (Tensor): 目标序列 token IDs，形状 (batch_size, tgt_seq_len)
            encoder_output (Tensor): 编码器输出，形状 (batch_size, src_seq_len, d_model)
            src_mask (Tensor): 源序列掩码，形状 (batch_size, 1, 1, src_len)
            tgt_mask (Tensor): 目标序列掩码，形状 (batch_size, 1, tgt_len, tgt_len)

        返回:
            Tensor: 解码器输出，形状 (batch_size, tgt_seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len=5000):
        """
        初始化 Transformer 模型

        参数:
            src_vocab_size (int): 源语言词汇表大小
            tgt_vocab_size (int): 目标语言词汇表大小

            d_model (int): 模型维度（词嵌入维度）
            num_layers (int): 编码器/解码器的层数
            num_heads (int): 多头注意力的头数

            d_ff (int): 前馈网络隐藏层维度

            dropout (float): Dropout 比率
            max_len (int): 最大序列长度，默认 5000
        """
        super(Transformer, self).__init__()
        # 编码器
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        # 解码器
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)

        # 线性层
        # input: d_model维， output: tgt_vocab_size维
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        """
        生成源序列和目标序列的掩码

        参数:
            src (Tensor): 源序列，形状 (batch_size, src_seq_len)
            tgt (Tensor): 目标序列，形状 (batch_size, tgt_seq_len)

        返回:
            tuple: (src_mask, tgt_mask)
                - src_mask: 源序列掩码，形状 (batch_size, 1, 1, src_len)
                - tgt_mask: 目标序列掩码，形状 (batch_size, 1, tgt_len, tgt_len)
        """
        # src_mask: (batch_size, 1, 1, src_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
        tgt_len = tgt.size(1)
        # 下三角矩阵，用于防止看到未来的 token
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).bool()  # (tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Transformer 前向传播

        参数:
            src (Tensor): 源序列 token IDs，形状 (batch_size, src_seq_len)
            tgt (Tensor): 目标序列 token IDs，形状 (batch_size, tgt_seq_len)

        返回:
            Tensor: 输出 logits，形状 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        output = self.final_linear(decoder_output)
        return output


# --- 演示如何使用模型 ---

def test_transformer():
    """
    测试 Transformer 模型的前向传播
    """
    # 1. 定义超参数
    src_vocab_size = 5000  # 源语言词汇表大小
    tgt_vocab_size = 5000  # 目标语言词汇表大小

    d_model = 512  # 模型维度（词嵌入维度）
    num_layers = 6  # 编码器/解码器层数
    num_heads = 8  # 注意力头数

    d_ff = 2048  # 前馈网络隐藏层维度（通常是 d_model 的 4 倍）

    dropout = 0.1  # Dropout 比率
    max_len = 100  # 最大序列长度

    # 2. 实例化模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)

    # 3. 创建模拟输入数据
    """
    src = torch.randint(1, src_vocab_size, (2, 10))
    
    解释：
    1: 随机数的起始值 (包含)
    src_vocab_size: 随机数的结束值 (不包含，通常是词表大小，如 10000)
    
    (2, 10): 生成数据的形状
        2: batch_size(批次大小),表示一次处理 2 个样本
        10: seq_length(序列长度),表示每个样本有 10 个词
    
    总结：生成了一个2行10列的张量，里面填充了 1 到 src_vocab_size-1 之间的随机整数
    
    比如：
    [[345, 7821, 1234, ..., 567, 890, 2345],
     [9876, 543, 2109, ..., 432, 765, 1098]]
    每一行代表一个句子的编码 (10 个单词的 ID),共 2 个句子。
    
    """
    src = torch.randint(1, src_vocab_size, (2, 10))
    tgt = torch.randint(1, tgt_vocab_size, (2, 12))

    # 4. 模型前向传播
    output = model(src, tgt)

    # 5. 打印输出形状
    print("模型输出的形状:", output.shape)
    # 预期输出: torch.Size([2, 12, 5000]) -> (batch_size, tgt_seq_len, tgt_vocab_size)


def _main():
    test_transformer()


if __name__ == "__main__":
    _main()
