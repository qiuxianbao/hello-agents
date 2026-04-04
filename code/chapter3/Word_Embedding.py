import numpy as np

"""
词嵌入

示例：语义的平移：我们从“国王”这个点出发，减去“男性”的向量，再加上“女性”的向量，最终就抵达了“女王”的位置
"""


def test_csine_similarity():
    # 假设我们已经学习到了简化的二维词向量
    embeddings = {
        "king": np.array([0.9, 0.8]),
        "queen": np.array([0.9, 0.2]),
        "man": np.array([0.7, 0.9]),
        "woman": np.array([0.7, 0.3])
    }

    # king - man + woman
    result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]
    print(result_vec)

    # 计算结果向量与 "queen" 的相似度
    sim = cosine_similarity(result_vec, embeddings["queen"])

    print(f"king - man + woman 的结果向量: {result_vec}")
    print(f"该结果与 'queen' 的相似度: {sim:.4f}")


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    余弦相似度公式:
    similarity(a⃗, b) = cos(θ) = (a⃗·b⃗) / (||a⃗|| × ||b⃗||)
    
    其中:
    - a·b⃗: 两个向量的点积 (dot product)
    - ||a⃗||: 向量 a 的模长 (L2 范数)
    - ||b⃗||: 向量 b 的模长 (L2 范数)
    
    几何意义:
    - 通过计算两个向量夹角的余弦值来衡量它们的相似度
    - 取值范围：[-1, 1],值越接近 1 表示越相似
    - 与向量的长度无关，只与方向有关

    返回:
        余弦相似度值 (范围 [-1, 1])
    """
    dot_product = np.dot(vec1, vec2)  # 点积
    print(f"点积: {dot_product}")

    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)  # 模长
    print(f"模长: {norm_product}")

    return dot_product / norm_product


def _main():
    test_csine_similarity()


if __name__ == '__main__':
    _main()
