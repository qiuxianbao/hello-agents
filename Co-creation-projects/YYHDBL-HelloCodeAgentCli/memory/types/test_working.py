import heapq
from datetime import datetime, timedelta

from hello_agents.memory import MemoryConfig
from hello_agents.memory import MemoryItem


class TestWorking:


    def test_(self):
        pass


    def test_tfidf(self):
        """
        TF-IDF (Term Frequency-Inverse Document Frequency)

        TF-IDF 是一种用于信息检索和文本挖掘的常用加权技术，
        用于评估一个词在一个文档集合中的重要程度。
        """

        # 导入必要的库
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # 示例文档集合
        documents = [
            "The cat sat on the mat",
            "The dog ran in the park",
            "Cats and dogs are pets",
            "I love my pet cat",
            "Running in park is fun"
        ]

        print("原始文档:")
        for i, doc in enumerate(documents):
            print(f"{i+1}. {doc}")

        # 创建TF-IDF向量化器
        """
        知识点：L2 归一化是一种将向量缩放到单位长度的技术（向量各元素平方和的平方根）
        L2范数：向量中各元素平方和的平方根，数学表示为 ||x||₂ = √(x₁² + x₂² + ... + xn²)
        L2距离：基于L2范数计算的空间中两点间的欧几里得距离
        """
        vectorizer = TfidfVectorizer(
            stop_words='english',    # 移除英文停用词 like 'the', 'is', 'at'
            lowercase=True,          # 转换为小写
            max_features=100         # 最多保留100个特征词
            # 默认norm = "l2"，会进行归一化处理
        )

        # 对文档进行TF-IDF向量化
        """
        vectorizer.fit_transform(documents) 的运算过程可以分解为以下几个步骤
        
        ## 运算流程
        ### 1. **fit()阶段**
        - **文本预处理**：根据参数设置对文档进行处理
          - 转换为小写（`lowercase=True`）
          - 移除停用词（`stop_words='english'`）
        - **构建词汇表**：扫描所有文档，提取唯一词汇并建立索引映射
        
        - **计算 IDF 值**：对每个词汇计算逆文档频率
          ```
          IDF(t) = log(文档总数 / 包含词t的文档数)
          ```
          
          ```
          注意：sklearn 中的 TfidfVectorizer 使用平滑IDF公式：
          IDF(t) = log((文档总数 + 1) / (包含词t的文档数 + 1)) + 1
          ```
          
          
        ### 2. **`transform()` 阶段**
        - **TF 计算**：对每个文档中的每个词计算词频
          ```
          TF(t,d) = 词t在文档d中的出现次数 / 文档d的总词数
          ```
        
        - **TF-IDF 计算**：组合 TF 和 IDF 得到最终权重
          ```
          TF-IDF(t,d) = TF(t,d) × IDF(t)
          ```
   
        ### 3. **结果输出**
        - **数据结构**：返回 `scipy.sparse.csr_matrix` 稀疏矩阵
        - **矩阵维度**：`(文档数量, 词汇表大小)`
        - **存储优化**：只存储非零值（因为大多数词不会出现在所有文档中）
        
        ### 4. **具体到代码中的参数影响**
        - `max_features=100`：限制词汇表最多100个词
        - `stop_words='english'`：移除常见词汇如 "the", "is", "at"
        - 最终得到形状为 `(5, 13)` 的矩阵（根据代码注释显示有13个特征词）
        
        这种运算方式使得相似的文档在向量空间中距离更近，为后续的相似度计算奠定基础。
        
        ##举个例子：
        文档1分析："The cat sat on the mat"
        经过预处理（移除停用词、小写化）后，文档1包含：["cat", "sat", "mat"]
        
        // 【词在多少个文档中出现过】
        1. 词汇统计
        cat: 出现在文档1、4 → 2个文档
        sat: 出现在文档1 → 1个文档
        mat: 出现在文档1 → 1个文档
        
        2. 平滑IDF计算（文档总数=5）
        IDF(cat) = log((5+1)/(2+1)) + 1 = log(2) + 1 ≈ 1.693
        IDF(sat) = log((5+1)/(1+1)) + 1 = log(3) + 1 ≈ 2.099
        IDF(mat) = log((5+1)/(1+1)) + 1 = log(3) + 1 ≈ 2.099
        
        // 【词在当前文档中出现的次数】
        3. TF计算（文档1总词数=4）
        TF(cat, 文档1) = 1/4 = 0.25
        TF(sat, 文档1) = 1/4 = 0.25
        TF(mat, 文档1) = 1/4 = 0.25
        
        // 【相乘】
        4. 原始TF-IDF值
        TF-IDF(cat) = 0.25 × 1.693 ≈ 0.423
        TF-IDF(sat) = 0.25 × 2.099 ≈ 0.525
        TF-IDF(mat) = 0.25 × 2.099 ≈ 0.525
        
        // 【L2归一化处理】
        5. L2归一化
        文档1向量: [0.423, 0.525, 0.525] (分别对应cat, sat, mat)
        L2范数: √(0.423² + 0.525² + 0.525²) = √(0.179 + 0.276 + 0.276) = √0.731 ≈ 0.855
        
        归一化后:
        cat: 0.423 / 0.855 ≈ 0.495
        sat: 0.525 / 0.855 ≈ 0.614
        mat: 0.525 / 0.855 ≈ 0.614
        
        6. 与实际值对比
        cat: 计算值0.495 ≈ 实际值0.4955 ✅
        sat: 计算值0.614 ≈ 实际值0.6142 ✅
        mat: 计算值0.614 ≈ 实际值0.6142 ✅
        """
        tfidf_matrix = vectorizer.fit_transform(documents)
        # print(type(tfidf_matrix))     # <class 'scipy.sparse._csr.csr_matrix'>

        """
        输出 5行13 列的【稀疏矩阵】
        行代表文档，列代表词汇
        
        <Compressed Sparse Row sparse matrix of dtype 'float64'
            with 15 stored elements and shape (5, 13)>
          Coords（坐标）	Values（值）
          (0, 0)	0.49552379079705033
          (0, 12)	0.6141889663426562
          (0, 6)	0.6141889663426562
          (1, 2)	0.6141889663426562
          (1, 10)	0.6141889663426562
          (1, 7)	0.49552379079705033
          (2, 1)	0.5773502691896258
          (2, 3)	0.5773502691896258
          (2, 9)	0.5773502691896258
          (3, 0)	0.49552379079705033
          (3, 5)	0.6141889663426562
          (3, 8)	0.6141889663426562
          (4, 7)	0.49552379079705033
          (4, 11)	0.6141889663426562
          (4, 4)	0.6141889663426562
        """
        # print(tfidf_matrix)

        # 获取特征名称（词汇表）
        feature_names = vectorizer.get_feature_names_out()

        # 词汇表 (13 个词):
        print(f"\n词汇表 ({len(feature_names)} 个词):")
        # ['cat', 'cats', 'dog', 'dogs', 'fun', 'love', 'mat', 'park', 'pet', 'pets', 'ran', 'running', 'sat']
        print(list(feature_names))

        # 显示TF-IDF矩阵形状
        # TF-IDF矩阵形状: (5, 13)
        print(f"\nTF-IDF矩阵形状: {tfidf_matrix.shape}")
        print("(行代表文档，列代表词汇)")

        # 转换为【稠密矩阵】以便查看
        dense_matrix = tfidf_matrix.toarray()

        """
        文档 1:
          'cat': 0.4955
          'mat': 0.6142
          'sat': 0.6142
        文档 2: 
          'dog': 0.6142
          'park': 0.4955
          'ran': 0.6142
        文档 3:
          'cats': 0.5774
          'dogs': 0.5774
          'pets': 0.5774
        文档 4:
          'cat': 0.4955
          'love': 0.6142
          'pet': 0.6142
        文档 5:
          'fun': 0.6142
          'park': 0.4955
          'running': 0.6142
        """
        print("\nTF-IDF 矩阵 (每行代表一个文档的TF-IDF向量):")
        for i, doc_vector in enumerate(dense_matrix):
            # np.nonzero(doc_vector)，返回数组中非零元素的索引数组，返回值是一个tuple
            # ([ 0  6 12],)，取元组的中第一个位置元素
            non_zero_indices = np.nonzero(doc_vector)[0]
            print(f"文档 {i+1}:")
            for idx in non_zero_indices:
                word = feature_names[idx]
                tfidf_score = doc_vector[idx]
                print(f"  '{word}': {tfidf_score:.4f}")

        """
        知识点：计算两个向量之间夹角的余弦值
        范围：[-1, 1]，值越接近1表示越相似
        特点: 只考虑向量方向，不考虑向量长度
        cos(θ) = (A·B) / (||A|| × ||B||)
        
        说明：
        A·B: 向量A和B的点积
        ||A|| × ||B||: 向量A和B的L2范数之积
        
        示例：
                    0                       6                      12
        文档1向量: [0.4955, 0, 0, 0, 0, 0, 0.6142, 0, 0, 0, 0, 0, 0.6142] (对应cat, mat, sat)
                    0                   5              8
        文档4向量: [0.4955, 0, 0, 0, 0, 0.6142, 0, 0, 0.6142, 0, 0, 0, 0] (对应cat, love, pet)
        
        
        点积计算: A·B = 0.4955×0.4955 + ... = 0.2455
        L2范数: ||A|| = √(0.4955² + 0.6142² + 0.6142²) ≈ 0.934
        L2范数: ||B|| = √(0.4955² + 0.6142² + 0.6142²) ≈ 0.934
        余弦相似度: 0.2455 / (0.934 × 0.934) ≈ 0.246
        
        #
        文档间相似度矩阵:
                Doc 1   Doc 2   Doc 3   Doc 4   Doc 5 
        Doc 1   1.000   0.000   0.000   0.246   0.000 
        Doc 2   0.000   1.000   0.000   0.000   0.246 
        Doc 3   0.000   0.000   1.000   0.000   0.000 
        Doc 4   0.246   0.000   0.000   1.000   0.000 
        Doc 5   0.000   0.246   0.000   0.000   1.000 
        """
        # 计算文档间的余弦相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)

        print("\n文档间相似度矩阵:")
        print("     ", end="")
        for j in range(len(documents)):
            # 格式化：2d是格式化字符串的一个格式说明符，用于数字的格式化输出
            # 2 代表的是 最小字段宽度为2个字符
            # d 代表的是 十进制整数（decimal integer）
            # 2d这意味着数字将至少占用2个字符的宽度，如果数字本身只有1位数，则会在前面补一个空格使其达到2个字符的宽度
            print(f"Doc{j+1:2d}", end=" ")
        print()

        for i, row in enumerate(similarity_matrix):
            print(f"Doc{i+1:2d}", end=" ")
            for sim_score in row:
                print(f"{sim_score:.3f}", end=" ")
            print()


    def test_tfidf_query(self):
        # 导入必要的库
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # 示例文档集合
        documents = [
            "The cat sat on the mat",
            "The dog ran in the park",
            "Cats and dogs are pets",
            "I love my pet cat",
            "Running in park is fun"
        ]

        vectorizer = TfidfVectorizer(
            stop_words='english',    # 移除英文停用词 like 'the', 'is', 'at'
            lowercase=True,          # 转换为小写
            max_features=100         # 最多保留100个特征词
        )

        # 对文档进行TF-IDF向量化
        tfidf_matrix = vectorizer.fit_transform(documents)
        # 获取特征名称（词汇表）
        feature_names = vectorizer.get_feature_names_out()

        # 词汇表 (13 个词):
        print(f"\n词汇表 ({len(feature_names)} 个词):")
        # ['cat', 'cats', 'dog', 'dogs', 'fun', 'love', 'mat', 'park', 'pet', 'pets', 'ran', 'running', 'sat']
        print(list(feature_names))

        # 显示TF-IDF矩阵形状
        # TF-IDF矩阵形状: (5, 13)
        print(f"\nTF-IDF矩阵形状: {tfidf_matrix.shape}")



        # 演示查询与文档的相似度计算
        query = "pet animals"
        print(f"\n查询: '{query}'")

        # 将查询转换为TF-IDF向量
        query_vector = vectorizer.transform([query])

        """
        查询转换为向量: <Compressed Sparse Row sparse matrix of dtype 'float64'
            with 1 stored elements and shape (1, 13)>
          Coords	Values
          (0, 8)	1.0
        """
        print(f"\n查询转换为向量: {query_vector}")

        # 计算查询与所有文档的相似度
        query_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        """
        查询与各文档的相似度:
          文档 1: 0.0000 - 'The cat sat on the mat'
          文档 2: 0.0000 - 'The dog ran in the park'
          文档 3: 0.0000 - 'Cats and dogs are pets'
          文档 4: 0.6142 - 'I love my pet cat'
          文档 5: 0.0000 - 'Running in park is fun'
        """
        print("查询与各文档的相似度:")
        for i, sim_score in enumerate(query_similarities):
            print(f"  文档 {i+1}: {sim_score:.4f} - '{documents[i]}'")

        # 按相似度排序
        ranked_docs = sorted(
            # 对于每个文档，创建一个包含三个元素的元组
            [(i, sim_score, documents[i]) for i, sim_score in enumerate(query_similarities)],
            # 指定排序依据为元组的第二个元素（即相似度得分）
            key=lambda x: x[1],
            # 降序排序
            reverse=True
        )

        """
        ranked_docs: [
        (3, np.float64(0.6141889663426563), 'I love my pet cat'), 
        (0, np.float64(0.0), 'The cat sat on the mat'), 
        (1, np.float64(0.0), 'The dog ran in the park'), 
        (2, np.float64(0.0), 'Cats and dogs are pets'), 
        (4, np.float64(0.0), 'Running in park is fun')]
        """
        print(f"\nranked_docs: {ranked_docs}")

        """
        按相似度排序的结果:
        1. 相似度: 0.6142, 文档: 'I love my pet cat'
        2. 相似度: 0.0000, 文档: 'The cat sat on the mat'
        3. 相似度: 0.0000, 文档: 'The dog ran in the park'
        4. 相似度: 0.0000, 文档: 'Cats and dogs are pets'
        5. 相似度: 0.0000, 文档: 'Running in park is fun'
        """
        print("\n按相似度排序的结果:")
        for rank, (idx, score, doc) in enumerate(ranked_docs, 1):
            print(f"{rank}. 相似度: {score:.4f}, 文档: '{doc}'")


    def test_float(self):
        f = float('inf')
        print(type(f)) ##<class 'float'>


    def test_getattr(self):
        config = MemoryConfig()

        max_age_minutes = getattr(config, 'working_memory_ttl_minutes', 120)
        print(max_age_minutes)
        # print(getattr(config, 'working_memory_ttl_minute', 100))  #100
        print(timedelta(minutes=max_age_minutes)) #2:00:00
        print(datetime.now() - timedelta(minutes=max_age_minutes))


    def test_heap(self):
        # 创建几个测试记忆项
        memory1 = MemoryItem(
            id="1",
            content="Important task",
            memory_type="working",
            user_id="test_user",
            timestamp=datetime.now(),
            importance=0.9
        )

        memory2 = MemoryItem(
            id="2",
            content="Less important task",
            memory_type="working",
            user_id="test_user",
            timestamp=datetime.now(),
            importance=0.5
        )

        memory3 = MemoryItem(
            id="3",
            content="Very important task",
            memory_type="working",
            user_id="test_user",
            timestamp=datetime.now(),
            importance=0.95
        )

        # 定义最小堆
        memory_heap = []

        # 计算优先级（在实际代码中会调用 _calculate_priority 方法）
        def calculate_priority(memory_item):
            return memory_item.importance  # 简化的优先级计算

        priority1 = calculate_priority(memory1)
        priority2 = calculate_priority(memory2)
        priority3 = calculate_priority(memory3)

        # 使用 heapq.heappush 添加记忆项到堆
        # 使用 -priority 是为了实现最大堆（优先级高的先取出）
        heapq.heappush(memory_heap, (-priority1, memory1.timestamp, memory1))
        print(f"Added memory1 with importance {memory1.importance} to heap")

        heapq.heappush(memory_heap, (-priority2, memory2.timestamp, memory2))
        print(f"Added memory2 with importance {memory2.importance} to heap")

        heapq.heappush(memory_heap, (-priority3, memory3.timestamp, memory3))
        print(f"Added memory3 with importance {memory3.importance} to heap")

        print(f"Heap size: {len(memory_heap)}")

        # 从堆中取出元素（优先级最高的会先被取出）
        print("\nRetrieving memories by priority:")
        while memory_heap:
            neg_priority, timestamp, memory_item = heapq.heappop(memory_heap)
            actual_priority = -neg_priority  # 转回正值
            # push时顺序是0.9,0.5,0.95；pop时顺序是0.95,0.9,0.5
            print(f"Retrieved memory '{memory_item.content}' with priority {actual_priority}")
