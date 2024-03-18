from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM

mpath = "zhongshsh/CLoT-cn"
tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(mpath, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    mpath, 
    device_map="cuda",
    trust_remote_code=True
).eval()

query = tokenizer.from_list_format([
    {'image': 'https://i.postimg.cc/Fz0bVzpm/test.png'},
    {'text': '让我们打破常规思维思考问题。请仔细阅读图片，写出一个令人感到意外且搞笑的句子。'},
])
response, history = model.chat(tokenizer, query=query, history=None, generation_config=generation_config)
print(response)
