import gradio as gr

def greet(name):
    return f"你好，{name}！"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    print(">>> Gradio 最小化测试启动")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
