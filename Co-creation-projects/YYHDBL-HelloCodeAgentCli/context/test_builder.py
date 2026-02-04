
class TestBuilder:

    def test_relevance_score(self):
        user_query = "This is a test query"

        packets = [
            "This is a packet",
            "This is a third test packet"
        ]

        query_tokens = set(user_query.lower().split())  # 默认按照空白字符进行拆分，包括空格、制表符、换行符等
        for i, packet in enumerate(packets, start=1):
            content_tokens = set(packet.lower().split())
            relevance_score = 0.0
            if len(query_tokens) > 0:
                # 计算2个集合的交集
                overlap = len(query_tokens & content_tokens)
                relevance_score = overlap / len(query_tokens)

            print(f"Packet{i}: Relevance score: {relevance_score}")



    def test_math(self):
        import math

        # 2.718281828459045
        print(math.exp(1))
