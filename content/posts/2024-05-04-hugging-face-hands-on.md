---
title: HuggingFace 学习笔记
date: 2024-05-04T12:19:27+08:00
tags: [HuggingFace, CodeCheatSheet]
categories: [学习笔记]
math: true
---

这部分的文档主要用作记录学习, 使用 HF 的时候常用的一些操作, 可能并不会很详细, 以及符合初学者的需求, 但是可以当做 CheatSheet 一类文档使用.

## 管道的使用

### 整体流程: 以情感分析为例

首先, 从 transformers 中导入 tokenizer, 主要将文本转成模型可以识别的 token:
```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

其次, 可以将文本数据先转化为可以识别的 token
```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

输出一个包含两个键的字典:
```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

使用模型 `AutoModel`:
```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

最终其输出为一个 _隐藏状态（hidden states）_，亦称 _特征(features)_
Transformers模块的矢量输出通常较大。它通常有三个维度：
- **Batch size**: 一次处理的序列数（在我们的示例中为2）。
- **Sequence length**: 序列的数值表示的长度（在我们的示例中为16）。
- **Hidden size**: 每个模型输入的向量维度。

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# 输出: torch.Size([2, 16, 768])
```

Transformers模型的输出直接发送到模型头进行处理:

{{< figure src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg" title="模型结构" >}}

有如下类型的 _头_
- `*Model` (retrieve the hidden states)
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`
- 以及其他 🤗

### 加载模型的方法

从头开始加载:
```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!

print(config)

# 输出:
# BertConfig {
#   [...]
#   "hidden_size": 768,
#   "intermediate_size": 3072,
#   "max_position_embeddings": 512,
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   [...]
# }
```

加载已经训练好的模型 (也可以使用 `AutoModel*` 类):
```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

保存模型:
```python
model.save_pretrained("directory_on_my_computer")

# terminal
# ls directory_on_my_computer
# terminal 输出:
# config.json pytorch_model.bin
```
### Tokenizer

它们有一个目的：将文本转换为模型可以处理的数据。模型只能处理数字，因此标记器(Tokenizer)需要将我们的文本输入转换为数字数据。

直接进行加载:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

使用:
```python
tokenizer("Using a Transformer network is simple")

# 输出
# {'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

进行编码的过程:
```python
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
# 输出:['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# 输出: [7993, 170, 11303, 1200, 2443, 1110, 3014]
```

解码:
```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
# 输出: 'Using a Transformer network is simple'
```


### 多个序列的处理方式
- 我们如何处理多个序列？
- 我们如何处理多个序列不同长度?
- 词汇索引是让模型正常工作的唯一输入吗？
- 是否存在序列太长的问题？

原则1: 模型需要一个 batch 的输出. 也就是说模型需要的输入形状是如下所示
```python
# Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
```

原则2: `attention_mask` 的作用
```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)

# 输出
# tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
# tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)

# 输出(得到相同的结果)
# tensor([[ 1.5694, -1.3895],
#         [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
```

原则3: 长序列需要截断
> 对于Transformers模型，我们可以通过模型的序列长度是有限的。大多数模型处理多达512或1024个令牌的序列，当要求处理更长的序列时，会崩溃。此问题有两种解决方案：
> - 使用支持的序列长度较长的模型。
> - 截断序列。

### 如何组合上述的使用

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

可以设置参数补足:
```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

可以设置参数截断:
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

可以设置参数返回特定类型:
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

直接使用 `Tokenizer()` (直接调用标记器(`Tokenizer`)对象), 其会根据模型添加 `[CLS]` 等特殊字符
### 微调
#### 准备好数据集

假如我们有一个 文本 数据集:
```python
from transformers import AutoTokenizer
from datasets import load_dataset

checkpoint = "bert-base-uncased"
tokenizer =  AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

# 输出:
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })

```

在上一节中我们知道应该如何将里面的 sentence 转化为 token:
```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

缺点是返回字典(字典的键是**输入词id(input_ids)**,  **注意力遮罩(attention_mask)** 和 **类型标记ID(token_type_ids)**, 字典的值是键所对应值的列表). 而且只有当您在转换过程中有**足够的内存**来存储整个数据集时才不会出错

所以使用下述方法:
```python
def tokenize_function(example):
    return (
	    tokenizer(example["sentence1"], 
				  example["sentence2"], truncation=True)
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

## 如何使用 `PEFT` 库

主要参考: 
- [自定义微调Demo](https://github.com/huggingface/peft/blob/main/examples/multilayer_perceptron/multilayer_perceptron_lora.ipynb)
- [官方文档](https://huggingface.co/docs/peft/en/developer_guides/custom_models)

为了使得调用的接口更加方便, 主要使用比较底层的接口, 下述为 training 的过程及参数:
```python
lr = 0.002
batch_size = 64
max_epochs = 30
device = "cpu" if not torch.cuda.is_available() else "cuda"

def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for xb, yb in eval_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                outputs = model(xb)
            loss = criterion(outputs, yb)
            eval_loss += loss.detach().float()

        eval_loss_total = (eval_loss / len(eval_dataloader)).item()
        train_loss_total = (train_loss / len(train_dataloader)).item()
        print(f"{epoch=:<2}  {train_loss_total=:.4f}  {eval_loss_total=:.4f}")
```

如果使用 `PEFT`:
```python
config = peft.LoraConfig(
    r=8,
    target_modules=["seq.0", "seq.2"], # 使用 LoRA 训练的层数, 参见 LoRA Config 的参数, 会使用regrex 匹配, https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
    modules_to_save=["seq.4"], # 未使用 LoRA 训练的层数
)

module = MLP().to(device)
module_copy = copy.deepcopy(module)  # we keep a copy of the original model for later
peft_model = peft.get_peft_model(module, config)
optimizer = torch.optim.Adam(peft_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
peft_model.print_trainable_parameters()

# 输出: trainable params: 56,164 || all params: 4,100,164 || trainable%: 1.369798866581922
```

查看每一层的名称, 对应 `config` 就可以知道哪些被训练了, 哪些未被训练, 根据输出, 可知只有很少一部分被训练了
```python
[(n, type(m)) for n, m in MLP().named_modules()]
# 输出:
[('', __main__.MLP),
 ('seq', torch.nn.modules.container.Sequential),
 ('seq.0', torch.nn.modules.linear.Linear),
 ('seq.1', torch.nn.modules.activation.ReLU),
 ('seq.2', torch.nn.modules.linear.Linear),
 ('seq.3', torch.nn.modules.activation.ReLU),
 ('seq.4', torch.nn.modules.linear.Linear),
 ('seq.5', torch.nn.modules.activation.LogSoftmax)]
```

训练过程:
```python
%time train(peft_model, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)
```

其实 `PEFT` model 和 `base` model 类似, `forward` 的接口和 torch 都是类似的. 根据此部分的内容可以直接适应到训练框架中.