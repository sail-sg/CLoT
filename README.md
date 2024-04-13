## Creative Leap-of-Thought (CLoT)
[![paper](https://img.shields.io/badge/cs.AI-2312.02439-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2312.02439)
[![page](https://img.shields.io/badge/Project_Page-CLoT-orange)](https://zhongshsh.github.io/CLoT/)
</br>


<p align="center">
  <img src="image/logo2.png" width="550" height="150"> 
</p>


This repository is the official codebase of "Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation" [[paper]](https://arxiv.org/abs/2312.02439). Our paper has been accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024 (CVPR 2024).

🤣👉**Click [[project page]](https://zhongshsh.github.io/CLoT/) for more funny examples**👈.



## 🤣 Introduction

To the best of our knowledge, we are the first to profoundly explore the Leap-of-Thought (LoT) ability in multimodal large language models (LLMs). This involves challenging LLMs to **think outside the box**, a non-sequential thinking skill equally crucial alongside popular sequential thinking abilities, such as Chain-of-Thought based methods. In this study, we delve into the LLM's LoT ability through the lens of a humor generation game called Oogiri (大喜利). The Oogiri game serves as an ideal platform for exploring the LLM's LoT ability, as it compels participants to think outside the box and provide unexpected and humorous responses to multimodal information (including I2T, T2T, and IT2T). 

<p align="center">
  <img src="image/example.png">
</p>

## 🤗 Quickstart

We provide a simple Chinese example in `inference.py` for using CLoT with zero-shot inference. Before you start, make sure you install the following packages:

```shell
pip install -r requirements.txt
```

Then run the command below:

```shell
python inference.py
```

## 💬 Gradio Web UI

Try launching the Gradio web interface with the following code!

```shell
python gradio_demo.py
```

## 😆 News

2024/4/13 - We released our [dataset](https://modelscope.cn/datasets/shan233/CLoT-Oogiri-GO/summary) and [checkpoint](https://modelscope.cn/models/shan233/CLoT-cn/summary) in ModelScope. 👈😆Please try it if you can't access Hugging Face! 

2024/3/16 - We released our [dataset](https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO) and [checkpoint](https://huggingface.co/zhongshsh/CLoT-cn). 👈😆Please try it! 

2023/12/6 - We released our [project page](https://zhongshsh.github.io/CLoT/). 👈😆Please check it out! 

2023/12/5 - We released our paper [[arxiv]](https://arxiv.org/abs/2312.02439). Please check it out! 


## 😂 TODO

- [x] project page
- [x] preprint paper
- [x] ~~online demo (Hugging Face etc.)~~ checkpoint
- [x] dataset
- [x] code


## 😄 Citation

```
@misc{zhong2023clot,
  title={Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation},
  author={Zhong, Shanshan and Huang, Zhongzhan and Gao, Shanghua and Wen, Weushao and Lin, Liang and Zitnik, Marinka and Zhou, Pan},
  journal={arXiv preprint arXiv:2312.02439},
  year={2023}
}
```
