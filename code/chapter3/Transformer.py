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
        # 计算 查询向量（Query） 和 键向量（Key） 的点积
        
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

            d_model (int): 模型维度（词嵌入维度），表示用多少维的向量表示一个token
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

"""
# 组件层次关系
Transformer
├── Encoder
│   ├── Embedding
│   ├── PositionalEncoding
│   └── EncoderLayer × N
│       ├── MultiHeadAttention
│       ├── PositionWiseFeedForward
│       └── LayerNorm × 2
├── Decoder
│   ├── Embedding
│   ├── PositionalEncoding
│   └── DecoderLayer × N
│       ├── MultiHeadAttention (self) × 2
│       ├── PositionWiseFeedForward
│       └── LayerNorm × 3
└── Linear (final)

#　时序图
┌─────────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│   Test      │  Transformer│  │ Encoder │  │ Decoder │  │EncoderLyr│  │DecoderLyr│
└──────┬──────┘  └─────┬────┘  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘
       │               │            │           │              │             │
       │ 1.创建实例     │            │           │              │             │
       ├──────────────>│            │           │              │             │
       │               │            │           │              │             │
       │               │2.初始化     │           │              │             │
       │               ├───────────>│           │              │             │
       │               │  -embedding│           │              │             │
       │               │  -pos_encoder          │              │             │
       │               │  -layers[] │           │              │             │
       │               │            │           │              │             │
       │               │3.初始化     │           │              │             │
       │               ├──────────────────────>│              │             │
       │               │  -embedding│           │              │             │
       │               │  -pos_encoder          │              │             │
       │               │  -layers[] │           │              │             │
       │               │            │           │              │             │
       │               │            │           │              │             │
       │ 4.前向传播     │            │           │              │             │
       ├──────────────>│            │           │              │             │
       │ model(src,tgt)│            │           │              │             │
       │               │            │           │              │             │
       │               │5.生成掩码   │           │              │             │
       │               │──┐         │           │              │             │
       │               │<─┘         │           │              │             │
       │               │            │           │              │             │
       │               │6.编码器forward         │              │             │
       │               ├───────────>│           │              │             │
       │               │ encoder(   │           │              │             │
       │               │   src,     │           │              │             │
       │               │   src_mask)│           │              │             │
       │               │            │           │              │             │
       │               │            │7.embedding │              │             │
       │               │            │──┐        │              │             │
       │               │            │<─┘        │              │             │
       │               │            │8.pos_encode│              │             │
       │               │            │──┐        │              │             │
       │               │            │<─┘        │              │             │
       │               │            │           │              │             │
       │               │            │9.遍历layers│              │             │
       │               │            ├────────────────────────>│             │
       │               │            │ for layer in layers:     │             │
       │               │            │   layer(x, mask)         │             │
       │               │            │           │              │             │
       │               │            │            │10.self_attn │             │
       │               │            │            │──────────┐  │             │
       │               │            │            │MultiHead │  │             │
       │               │            │            │Attention │  │             │
       │               │            │            │<─────────┘  │             │
       │               │            │            │             │             │
       │               │            │            │11.feed_forward            │
       │               │            │            │──────────┐  │             │
       │               │            │            │PositionWise│             │
       │               │            │            │FeedForward│             │
       │               │            │            │<──────────┘ │             │
       │               │            │<─────────────────────────┘             │
       │               │            │           │              │             │
       │               │            │12.LayerNorm│              │             │
       │               │            │──┐        │              │             │
       │               │            │<─┘        │              │             │
       │               │<───────────┤           │              │             │
       │               │encoder_out │           │              │             │
       │               │            │           │              │             │
       │               │13.解码器forward         │              │             │
       │               ├──────────────────────> │              │             │
       │               │ decoder(   │           │              │             │
       │               │   tgt,     │           │              │             │
       │               │   enc_out, │           │              │             │
       │               │   masks)   │           │              │             │
       │               │            │           │              │             │
       │               │            │           │14.embedding  │             │
       │               │            │           │──┐           │             │
       │               │            │           │<─┘           │             │
       │               │            │           │15.pos_encode │             │
       │               │            │           │──┐           │             │
       │               │            │           │<─┘           │             │
       │               │            │           │              │             │
       │               │            │           │16.遍历layers  │             │
       │               │            │           ├────────────────────────>│
       │               │            │           │ for layer in layers:     │
       │               │            │           │   layer(...)             │
       │               │            │           │              │             │
       │               │            │           │              │17.masked_self_attn
       │               │            │           │              │──────────┐
       │               │            │           │              │MultiHead │
       │               │            │           │              │Attention │
       │               │            │           │              │<─────────┘
       │               │            │           │              │             │
       │               │            │           │              │18.cross_attn
       │               │            │           │              │──────────┐
       │               │            │           │              │MultiHead │
       │               │            │           │              │Attention │
       │               │            │           │              │<─────────┘
       │               │            │           │              │             │
       │               │            │           │              │19.feed_forward
       │               │            │           │              │──────────┐
       │               │            │           │              │PositionWise│
       │               │            │           │              │FeedForward│
       │               │            │           │              │<──────────┘
       │               │            │           │<─────────────────────────┘
       │               │            │           │              │             │
       │               │            │           │20.LayerNorm  │             │
       │               │            │           │──┐           │             │
       │               │            │           │<─┘           │             │
       │               │<───────────────────────┤              │             │
       │               │decoder_out │           │              │             │
       │               │            │           │              │             │
       │               │21.final_linear         │              │             │
       │               │──┐         │           │              │             │
       │               │<─┘         │           │              │             │
       │               │            │           │              │             │
       │<──────────────┤            │           │              │             │
       │ output(logits)│            │           │              │             │
       │               │            │           │              │             │

"""


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
    """
    这里的 model(...) 看起来像是在调用一个函数，但实际上是调用了 Transformer 对象的 forward 方法
    因为 Transformer 类继承自 nn.Module（PyTorch 的神经网络基类）
    
    在 PyTorch 中，当你继承 nn.Module 后：
    直接调用对象 model(src, tgt) 等价于 model.forward(src, tgt)
    
    """
    output = model(src, tgt)

    # 5. 打印输出形状
    print("模型输出的形状:", output.shape)
    # 预期输出: torch.Size([2, 12, 5000]) -> (batch_size, tgt_seq_len, tgt_vocab_size)


def test_matmul():
    """
    不同维度的作用
    :return:
    """
    print("=" * 60)
    print("实验1: 单样本单头")
    print("=" * 60)
    Q1 = torch.randn(10, 64)  # (seq_q, d_k)
    K_T1 = torch.randn(64, 10)  # (d_k, seq_k)

    result1 = torch.matmul(Q1, K_T1)
    print(f"Q1 shape: {Q1.shape}")
    print(f"K_T1 shape: {K_T1.shape}")
    print(f"result1 shape: {result1.shape}")  # (10, 10)

    print("\n" + "=" * 60)
    print("实验2: 单样本多头（8个头）")
    print("=" * 60)
    Q2 = torch.randn(8, 10, 64)  # (heads, seq_q, d_k)
    K_T2 = torch.randn(8, 64, 10)  # (heads, d_k, seq_k)

    result2 = torch.matmul(Q2, K_T2)
    print(f"Q2 shape: {Q2.shape}")
    print(f"K_T2 shape: {K_T2.shape}")
    print(f"result2 shape: {result2.shape}")  # (8, 10, 10)
    print("→ 对每个 head 分别做矩阵乘法")

    print("\n" + "=" * 60)
    print("实验3: 多样本多头（2个样本，8个头）")
    print("=" * 60)
    Q3 = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_q, d_k)
    K_T3 = torch.randn(2, 8, 64, 10)  # (batch, heads, d_k, seq_k)

    result3 = torch.matmul(Q3, K_T3)
    print(f"Q3 shape: {Q3.shape}")
    print(f"K_T3 shape: {K_T3.shape}")
    print(f"result3 shape: {result3.shape}")  # (2, 8, 10, 10)
    print("→ 对每个 batch 的每个 head 分别做矩阵乘法")

    print("\n" + "=" * 60)
    print("验证：手动拆解")
    print("=" * 60)

    # 手动验证第0个batch、第0个head的结果
    manual_result = torch.matmul(Q3[0, 0], K_T3[0, 0])
    auto_result = result3[0, 0]

    print(f"手动计算 result3[0,0] shape: {manual_result.shape}")
    print(f"自动计算 result3[0,0] shape: {auto_result.shape}")
    print(f"两者是否相等: {torch.allclose(manual_result, auto_result)}")  # True


def test_transpose():
    """
    为什么要转置 K 并做矩阵乘法？

    1. 数学本质：计算注意力得分
    在注意力机制中，我们需要计算 Query（查询） 和 Key（键） 之间的相似度分数。
    注意力得分 = Q · K^T
    这本质上是在计算每个 query 向量与每个 key 向量的点积。


    2. 维度分析
    假设我们有以下维度：
    Q: (batch_size, num_heads, seq_len_q, d_k)
    例如：(2, 8, 10, 64) → 2个样本，8个头，查询序列长度10，每个头维度64

    K: (batch_size, num_heads, seq_len_k, d_k)
    例如：(2, 8, 10, 64) → 同样的结构

    问题： 我们想要得到每个 query 与每个 key 的相似度，结果应该是：
    期望输出: (batch_size, num_heads, seq_len_q, seq_len_k)
    例如：(2, 8, 10, 10) → 10个query × 10个key 的得分矩

    3. 为什么需要转置？

    ```
    # Q: (2, 8, 10, 64)
    # K: (2, 8, 10, 64)
    torch.matmul(Q, K)  # ❌ 维度不匹配！
    ```

    矩阵乘法要求：(..., m, n) × (..., n, p) → (..., m, p)
    但这里是 (..., 10, 64) × (..., 10, 64)，最后两个维度不匹配（64 ≠ 10）。

    转置后的情况（正确）：
    # Q: (2, 8, 10, 64)
    # K.transpose(-2, -1): (2, 8, 64, 10)  ← 最后两维交换
    torch.matmul(Q, K.transpose(-2, -1))
    # ✅ (2, 8, 10, 64) × (2, 8, 64, 10) → (2, 8, 10, 10)

    transpose(-2, -1) 的含义：
    -2 表示倒数第二个维度（seq_len_k）
    -1 表示最后一个维度（d_k）
    转置后：(batch, heads, d_k, seq_len_k)

    4.直观理解：点积计算
    对于单个样本、单个头的情况：
    Q 的形状: (seq_len_q, d_k) = (10, 64)
       [q1]  ← 64维向量
       [q2]
       ...
       [q10]

    K 的形状: (seq_len_k, d_k) = (10, 64)
       [k1]
       [k2]
       ...
       [k10]

    K^T 的形状: (d_k, seq_len_k) = (64, 10)
       [k1, k2, ..., k10]  ← 每列是一个key向量

    矩阵乘法 Q · K^T：
            k1    k2    ...   k10
          ┌─────┬─────┬─────┬─────┐
     q1   │q1·k1│q1·k2│ ... │q1·k10│  ← 第1个query与所有key的相似度
          ├─────┼─────┼─────┼─────┤
     q2   │q2·k1│q2·k2│ ... │q2·k10│
          ├─────┼─────┼─────┼─────┤
     ...  │ ... │ ... │ ... │ ... │
          ├─────┼─────┼─────┼─────┤
     q10  │q10·k1│... │ ... │q10·k10│
          └─────┴─────┴─────┴─────┘

    结果形状: (10, 10) → 每个元素是一个点积得分

    示例：
    [1]       [4 5 6]   [1×4  1×5  1×6]   [4   5   6 ]
    [2]   ×           = [2×4  2×5  2×6] = [8   10  12]
    [3]                 [3×4  3×5  3×6]   [12  15  18]


    5. 为什么用点积？
    点积的物理意义：
    值越大 → 两个向量方向越接近 → 相似度越高
    值越小 → 两个向量方向差异越大 → 相似度越低

    在注意力机制中：
    q_i · k_j 表示：第 i 个位置的词对第 j 个位置的词的关注程度
    经过 softmax 后，变成概率分布（注意力权重）

    """
    # 输入
    """
    # 参数含义：
    batch_size, 【批次大小】, 比如：一次处理 2个样本（如2个句子）
    num_heads, 【注意力头数】, 比如：8个注意力头并行计算
    seq_len, 【序列长度】, 比如：每个样本有 10个词/token
    d_k, 【每个头的维度】, 比如：每个注意力头的向量维度是 64维
    
    # 4维度张量结构分析：
    torch.randn(2, 8, 10, 64)
    
    📦 最外层 (2): 2个句子
       └─ 📦 第2层 (8): 每个句子用8个头同时分析
          └─ 📦 第3层 (10): 每个头关注10个位置
             └─ 📦 最内层 (64): 每个位置用64维向量表示
    
    为什么是 64 维？
    在 Transformer 中：
    总维度 d_model = 512（常见配置）
    注意力头数 num_heads = 8
    每个头的维度 d_k = d_model / num_heads = 512 / 8 = 64
    
    这样设计的好处：
    1.并行计算：8个头可以同时处理不同的语义关系
    2.降低计算量：每个头只处理64维，而不是512维
    3.多视角学习：不同头可以关注不同的语言特征（语法、语义、指代等）
    
    总结：
    (batch, heads, seq_len, d_k)
      ↑       ↑       ↑       ↑
     几个    几组    几个    多长
     样本    视角    位置    向量
    
    """
    Q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_q, d_k[ dimension of key])
    K = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_k, d_k)

    # 步骤1: 转置 K
    K_T = K.transpose(-2, -1)  # (2, 8, 64, 10)

    # 步骤2: 矩阵乘法
    """
    torch.matmul 对于高维张量有一个特殊的规则：
    对于形状为 (..., m, n) 和 (..., n, p) 的两个张量
    matmul 会对前面的所有维度进行"广播"，只对最后两个维度做矩阵乘法
    
    结果形状: (..., m, p)
    
    
    # 内部发生了什么？
    torch.matmul 实际上是在做 2 × 8 = 16 次独立的矩阵乘法：
    # 伪代码展示内部逻辑
    for i in range(2):        # 遍历 batch
        for j in range(8):    # 遍历 heads
            # 每次取出一个 (10, 64) 和 (64, 10) 的矩阵相乘
            result[i, j] = Q[i, j] @ K_T[i, j]
            #               (10,64) @ (64,10) → (10,10)
    
    底层使用 GPU 并行（底层 C++/CUDA 实现）
    
    # 完整结构：
    Q 的形状: (2, 8, 10, 64)
    
    📦 Batch 0 (第1个句子)
       📦 Head 0: (10, 64) 矩阵  ──┐
       📦 Head 1: (10, 64) 矩阵    │
       📦 Head 2: (10, 64) 矩阵    │  分别与对应的 K_T 相乘
       ...                         │
       📦 Head 7: (10, 64) 矩阵  ──┘
    
    📦 Batch 1 (第2个句子)
       📦 Head 0: (10, 64) 矩阵  ──┐
       📦 Head 1: (10, 64) 矩阵    │
       ...                         │
       📦 Head 7: (10, 64) 矩阵  ──┘
    
    总共: 2 × 8 = 16 次独立的 (10,64) @ (64,10) → (10,10) 运算

    """
    attn_scores = torch.matmul(Q, K_T)  # (2, 8, 10, 10)

    # 步骤3: 缩放（防止梯度消失）
    attn_scores = attn_scores / math.sqrt(64)  # 除以 √d_k

    # 步骤4: Softmax 归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)  # (2, 8, 10, 10)

    # 步骤5: 加权求和
    V = torch.randn(2, 8, 10, 64)
    output = torch.matmul(attn_weights, V)  # (2, 8, 10, 64)


def _main():
    # test_transformer()
    test_matmul()
    # test_transpose()


if __name__ == "__main__":
    _main()
