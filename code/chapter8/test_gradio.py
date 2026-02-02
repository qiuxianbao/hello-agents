import gradio as gr

class TestGradio:

    def test_ui(self):
        def process_input(user_input):
            # 简单处理用户输入，返回结果
            if user_input.strip():
                return f"您输入的内容是：{user_input}"
            else:
                return "输入框为空，请输入一些内容"

        # 创建块
        with gr.Blocks(title="双标签页Demo") as demo:
            # 第1个标签
            with gr.Tab("标签页1"):
                # 创建2级标题
                gr.Markdown("## 第一个标签页")
                # 创建输入框、按钮
                input_text = gr.Textbox(label="输入框", placeholder="请输入内容...")
                submit_btn = gr.Button("提交按钮", variant="primary")
                output_result = gr.Textbox(label="输出结果", interactive=False)
                
                submit_btn.click(
                    fn=process_input,
                    inputs=input_text,
                    outputs=output_result
                )
                
                # 也可以支持回车提交
                input_text.submit(
                    fn=process_input,
                    inputs=input_text,
                    outputs=output_result
                )

            # 第2个标签
            with gr.Tab("标签页2"):
                gr.Markdown("## 第二个标签页")
                gr.Markdown("### 随机功能展示")
                
                # 添加一些有趣的组件
                name = gr.Textbox(label="姓名", placeholder="输入您的姓名")
                age = gr.Slider(1, 100, label="年龄", value=25)
                hobby = gr.CheckboxGroup([
                    "编程", "阅读", "运动", "音乐", "旅行"
                ], label="兴趣爱好")
                
                # 计算函数
                def greet(name, age, hobbies):
                    if name.strip():
                        hobby_str = ", ".join(hobbies) if hobbies else "无"
                        return f"你好 {name}！年龄：{age}岁，兴趣爱好：{hobby_str}"
                    else:
                        return "请先输入您的姓名"
                
                btn_greet = gr.Button("打招呼")
                output_greet = gr.Textbox(label="问候结果", interactive=False)
                
                btn_greet.click(
                    fn=greet,
                    inputs=[name, age, hobby],
                    outputs=output_greet
                )
        
        return demo


if __name__ == "__main__":
    # 创建TestGradio实例并获取UI
    app = TestGradio()
    demo = app.test_ui()

    # 启动Gradio应用
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
