# Huggingface

## transformers基本使用

包含模型加载，分词器加载，利用Huggingface上开源的预训练模型进行推理。

## datasets基本使用

利用datasets库处理训练所需的数据集。

# Pretraining

## tranformer

包括transfomer的基本架构，利用English-French dataset进行训练。

## Bert

包括bert的基本架构，利用wiki dataset进行训练。

# Finetuning

## TASK 1：关键字提取

任务描述：

利用Qwen开源仓库中微调代码，实现自定义的关键字提取任务。

实现思路：

从用户提问中提取日期和城市
调用天气预报的API，输入日期和城市，获取天气信息
输出给用户。

输入：任意一段包含关键字的输入

输出：关键字（常量）：变量

例子：

```
Q:2024年7月9号，嘉兴天气怎么样？
A:
日期：2024/07/09
城市：嘉兴
```

## Task 2: 风格微调

让LLM的输出是诗词的风格。

利用唐诗数据集微调LLM，使得LLM可以续写唐诗。

## Task 3: 分类模型微调



## Task 4: 图像风格微调



# Posttraining

## DPO

### DPO for Safety Alignment

利用tranformers提供的trl库，实现LLM的安全对齐。

### DPO for Preference Alignment

利用tranformers提供的trl库，实现LLM的偏好对齐。

# Prompt Engineer

## RAG 

利用langchain实现的简单RAG demo。

# 🙏Acknowledgements 🙏

我衷心感谢以下开源库。

[1] https://github.com/huggingface/huggingface_hub

[2] https://github.com/d2l-ai/d2l-zh

[3] https://github.com/owenliang/qwen-sft

[4] https://github.com/eric-mitchell/direct-preference-optimization

[5] https://github.com/langchain-ai/rag-from-scratch