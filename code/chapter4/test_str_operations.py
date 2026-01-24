def test_print_function():
    """测试字符串操作功能"""
    # str
    result = ["a", "b", "c"]
    
    # map
    results = {
        "answer_box_list": ["直接答案1", "直接答案2"],     # 可选：直接答案列表
        "answer_box": {                                # 可选：答案框
            "answer": "这是一个直接答案"
        },
        "knowledge_graph": {                          # 可选：知识图谱
            "description": "实体描述信息"
        },
        "organic_results": [                          # 原始搜索结果列表
            {
                "title": "搜索结果标题1",
                "snippet": "这是第一个搜索结果的摘要信息...",
                "link": "https://example1.com"
            },
            {
                "title": "搜索结果标题2",
                "snippet": "这是第二个搜索结果的摘要信息...",
                "link": "https://example2.com"
            },
            {
                "title": "搜索结果标题3",
                "snippet": "这是第三个搜索结果的摘要信息...",
                "link": "https://example3.com"
            }
        ]
    }

    # 获取前3条，使用enumerate获取索引
    snippets = [
        f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
        for i, res in enumerate(results["organic_results"][:3])
    ]

    # 测试输出132
    output_value = 132
    print(output_value)  # 这个print在pytest -s下可见
    
    # 验证结果
    assert len(result) == 3
    assert result[0] == "a"
    assert len(results["organic_results"]) == 3
    assert output_value == 132