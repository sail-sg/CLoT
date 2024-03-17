"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser

import gradio as gr
import os
import re
import glob
import torch
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM

DEFAULT_CKPT_PATH = "zhongshsh/CLoT-cn"
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = (
    "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
)
block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        default=True,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=2333, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path, device_map="cuda", trust_remote_code=True, fp16=True
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    return model, tokenizer


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args):
    model, tokenizer = _load_model_tokenizer(args.checkpoint_path)

    def predict(_chatbot, task_history):
        if len(task_history) == 0 or len(task_history[-1]) != 2:
            return _chatbot

        textbox, imagebox = task_history[-1]
        base_prompt = """
你是一个搞笑专家，非常了解网络和现实生活当中的各种梗和流行语，热衷于用幽默感给大家带来欢乐。让我们打破常规思维思考问题。

用户： """.strip()
        last_prompt = "回复: 符合要求的回复是"

        all_result = []
        if textbox == None:
            query_list = [
                "请仔细阅读图片，写出一个令人感到意外且搞笑的句子。",
                "根据图片，想出一个让人感到出乎意料和滑稽句子。",
                "请对这张图片仔细审视，创造一个出乎意料又滑稽可笑的句子。",
            ]
            for query in query_list:
                message = f"{base_prompt}{query}\nPicture 1: <img>{imagebox}</img>\n{last_prompt}\n"
                for _ in range(3):
                    response, history = model.chat(
                        tokenizer,
                        message,
                        history=[],
                        generation_config=model.generation_config,
                    )
                    chat_response = response.replace("<ref>", "")
                    chat_response = chat_response.replace(r"</ref>", "")
                    chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)
                    all_result.append(chat_response)

        elif imagebox == None:
            query_list = [
                "请仔细理解所给的问题，想出一个令人感到意外且搞笑的回答。",
                "请对所提供的问题进行仔细解读，给出一个出人意料且滑稽的答案。",
                "请细心品味所给问题，以构思一个意外而滑稽的答复。",
            ]
            for query in query_list:
                message = f"{base_prompt}{query}\n问题: {textbox}\n{last_prompt}\n"
                for _ in range(3):
                    response, history = model.chat(
                        tokenizer,
                        message,
                        history=[],
                        generation_config=model.generation_config,
                    )
                    chat_response = response.replace("<ref>", "")
                    chat_response = chat_response.replace(r"</ref>", "")
                    chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)
                    all_result.append(chat_response)

        else:
            query_list = [
                "请仔细阅读图片和图片里的文字，写出一个令人感到意外且搞笑的句子。",
                "请仔细阅读图片和图片里的文字，为图中指定部分补充令人感到意外且搞笑的句子。",
                "请细心品味图片的内容，为图中指定部分给出有趣且搞笑的说法。",
            ]
            for query in query_list:
                message = f"{base_prompt}{query}\nPicture 1: <img>{imagebox}</img>\n问题: {textbox}\n{last_prompt}\n"
                for _ in range(3):
                    response, history = model.chat(
                        tokenizer,
                        message,
                        history=[],
                        generation_config=model.generation_config,
                    )
                    chat_response = response.replace("<ref>", "")
                    chat_response = chat_response.replace(r"</ref>", "")
                    chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)
                    all_result.append(chat_response)

        all_result = list(set(all_result))
        all_result_str = ""
        for i in range(len(all_result)):
            all_result_str += f"{chr(ord('A')+i)}. {all_result[i]}\n"

        response = all_result_str
        chat_response = response.replace("<ref>", "")
        chat_response = chat_response.replace(r"</ref>", "")
        chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)

        if chat_response != "":
            _chatbot.append((None, chat_response))

        full_response = _parse_text(response)
        task_history.append(full_response)
        return _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history or len(task_history) <= 1:
            return _chatbot
        task_history = task_history[:-1]
        _chatbot.pop(-1)
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        if text == "":
            return history, task_history
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history

    def add_file(history, task_history, file):
        if file == None:
            return history, task_history

        # https://github.com/gradio-app/gradio/issues/3728
        history = history + [((file,), None)]
        if len(task_history) > 0 and task_history[-1][-1] == None:
            task_history[-1] = (task_history[-1][0], file)
        else:
            task_history = task_history + [(None, file)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_image_input():
        return None

    def reset_state(task_history):
        task_history.clear()
        return []

    def change_model(model_selector):
        global model, tokenizer
        torch.cuda.empty_cache()
        print("change to ", model_selector)
        model, tokenizer = _load_model_tokenizer(model_selector)
        torch.cuda.empty_cache()
        return []

    models = [args.checkpoint_path]

    textbox = gr.Textbox(
        show_label=False,
        placeholder="Enter question or only submit image",
        container=False,
    )
    with gr.Blocks(title="CLoT", theme=gr.themes.Default(), css=block_css) as demo:
        gr.Markdown(
            """<center><img src="https://i.postimg.cc/65hhZNyY/logo2.png" style="height: 160px"/></center>"""
        )
        gr.Markdown(
            """\
<center><font size=3>This WebUI is built upon <a href="https://huggingface.co/zhongshsh/CLoT-cn">Qwen-VL-CLoT (cn)</a> to achieve the generation of multimodal humorous content. \
(本WebUI基于Qwen-VL-CLoT打造，实现多模态幽默内容的生成)</center>"""
        )
        gr.Markdown(
            """\
<center><font size=3><a href="https://zhongshsh.github.io/CLoT">Project Page</a> 
| <a href="https://github.com/sail-sg/CLoT">Code</a> 
| <a href="https://huggingface.co/zhongshsh/CLoT-cn">🤗 Model</a> 
| <a href="https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO">🤗 Dataset</a></center>"""
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=args.checkpoint_path,
                        interactive=True,
                        show_label=False,
                        container=False,
                    )

                imagebox = gr.Image(type="filepath", height=300)
                textbox.render()
                submit_btn = gr.Button(
                    value="Let's think outside the box !", variant="primary"
                )
                gr.Examples(
                    examples=[
                        "最开心的事情是什么？",
                        "你曾经经历过的最尴尬的时刻是什么？",
                    ],
                    inputs=textbox,
                )

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                    examples=[
                        "https://i.postimg.cc/VNV5m0fh/test1.jpg",
                        "https://i.postimg.cc/4NQdnS4V/test2.jpg",
                        "https://i.postimg.cc/cJr491RB/test3.png",
                        "https://i.postimg.cc/C1n51Mdx/test4.png",
                        "https://i.postimg.cc/8zyzmf6C/test5.png",
                        "https://i.postimg.cc/0NF5LgZR/test6.png",
                        "https://i.postimg.cc/qMTqJTrp/test7.jpg",
                        "https://i.postimg.cc/2jv3N5PH/test8.jpg",
                        "https://i.postimg.cc/YC2SFzjV/test9.jpg",
                        "https://i.postimg.cc/L5zstytd/test10.jpg",
                    ],
                    inputs=imagebox,
                    examples_per_page=5,
                )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="CLoT Chatbot", height=650
                )

                with gr.Row(elem_id="buttons") as button_row:
                    regen_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    empty_btn = gr.Button(value="🗑️  Clear", interactive=True)

        task_history = gr.State([])
        submit_btn.click(
            add_text, [chatbot, task_history, textbox], [chatbot, task_history]
        ).then(
            add_file, [chatbot, task_history, imagebox], [chatbot, task_history]
        ).then(
            predict, [chatbot, task_history], [chatbot]
        ).then(
            reset_user_input, [], [textbox]
        ).then(
            reset_image_input, [], [imagebox]
        )

        empty_btn.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )
        model_selector.change(change_model, [model_selector], [])

        gr.Markdown(
            """\
<font size=2>Note: This demo is governed by the original license of Qwen-VL and CLoT. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen-VL和CLoT的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
        )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == "__main__":
    main()
