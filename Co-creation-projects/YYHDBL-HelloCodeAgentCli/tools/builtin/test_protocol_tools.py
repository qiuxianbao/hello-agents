import asyncio
import concurrent
import time


class TestProtocolTools:


    def task(self, n):
        """
        类方法
        需要通过类实例调用，或者以静态方法/普通函数的形式调用
        :param n:
        :return:
        """
        print(f"任务 {n} 开始执行\n")
        time.sleep(1)  # 模拟耗时操作
        print(f"任务 {n} 执行完成\n")
        # 返回
        return f"任务 {n} 的结果\n"

    def test_executor(self):

        # 创建线程池，最大线程数为3
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 提交任务到线程池
            # submit 方法会自动将参数传递给目标函数
            futures = [executor.submit(self.task, i) for i in range(5)]

            # 获取任务结果
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(result)


    def test_executor1(self):

        """
        嵌套
        属于局部函数
        :return:
        """
        def task(n):
            """模拟一个耗时任务"""
            print(f"任务 {n} 开始执行\n")
            time.sleep(1)  # 模拟耗时操作
            print(f"任务 {n} 执行完成\n")
            # 返回
            return f"任务 {n} 的结果\n"

        # 创建线程池，最大线程数为3
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 提交任务到线程池
            # submit 方法会自动将参数传递给目标函数
            futures = [executor.submit(task, i) for i in range(5)]

            # 获取任务结果
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(result)


    def test_async(self):
        """
        测试异步
        :return:
        """
        async def main():
            print('hello')

        try:
            asyncio.get_event_loop()

        except RuntimeError:
            asyncio.run(main())
