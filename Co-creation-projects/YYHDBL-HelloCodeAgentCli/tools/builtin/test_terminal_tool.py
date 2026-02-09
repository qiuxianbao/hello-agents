
class TestTerminalTool:


    def test_input(self):
        command = "rm -rf /"
        ans = input(f"\n⚠️ 高风险命令：{command}\n允许执行？(y/n)\nconfirm> ").strip().lower()
        if ans not in {"y", "yes"}:
            print("⛔️ 已取消执行（用户未确认）")
        else:
            print("🚀 正在执行...")

if __name__ == "__main__":
    TestTerminalTool().test_input()