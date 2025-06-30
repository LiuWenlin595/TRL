# 监督微调训练器

[![](https://img.shields.io/badge/All_models-SFT-blue)](https://huggingface.co/models?other=sft,trl) [![](https://img.shields.io/badge/smol_course-Chapter_1-yellow)](https://github.com/huggingface/smol-course/tree/main/1_instruction_tuning)

监督微调（SFT）是后训练基础模型中最常见的步骤，也是最有效的步骤之一。在TRL中，我们提供了一个简单的API，可以用几行代码训练SFT模型；对于完整的训练脚本，请查看[`trl/scripts/sft.py`](https://github.com/huggingface/trl/tree/main/trl/scripts/sft.py)。视觉语言模型的实验性支持也包含在[`examples/scripts/sft_vlm.py`](https://github.com/huggingface/trl/tree/main/examples/scripts/sft_vlm.py)中。

## 快速开始

如果您有一个托管在🤗 Hub上的数据集，您可以使用TRL的[`SFTTrainer`]轻松微调您的SFT模型。假设您的数据集是`imdb`，您想要预测的文本在数据集的`text`字段中，您想要微调`facebook/opt-350m`模型。
以下代码片段为您处理所有的数据预处理和训练：

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_length=512,
    output_dir="/tmp",
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```
确保为`max_length`传递正确的值，因为默认值将设置为`min(tokenizer.model_max_length, 1024)`。

您也可以在训练器外部构建模型并按如下方式传递：

```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

training_args = SFTConfig(output_dir="/tmp")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

上述代码片段将使用[`SFTConfig`]类中的默认训练参数。如果您想修改默认值，请将您的修改传递给`SFTConfig`构造函数，并通过`args`参数将其传递给训练器。

## 高级用法

### 仅在完成结果上训练

要仅在完成结果上训练，只需使用[提示-完成](dataset_formats#prompt-completion)数据集。在此模式下，损失仅计算在完成部分。

如果您想在提示**和**完成结果上都计算损失，同时仍使用提示-完成数据集，请在[`SFTConfig`]中设置`completion_only_loss=False`。这相当于[将数据集转换为语言建模](dataset_formats#from-prompt-completion-to-language-modeling-dataset)格式。

### 为聊天格式添加特殊标记

向语言模型添加特殊标记对于训练聊天模型至关重要。这些标记添加在对话中不同角色之间，如用户、助手和系统，帮助模型识别对话的结构和流程。这种设置对于使模型能够在聊天环境中生成连贯且上下文适当的响应至关重要。
[`clone_chat_template`]函数是一个有用的工具，用于为对话AI任务准备模型和分词器。此函数：
- 向分词器添加特殊标记，例如`<|im_start|>`和`<|im_end|>`，以指示对话的开始和结束。
- 调整模型的嵌入层以适应新标记。
- 设置分词器的`chat_template`，用于将输入数据格式化为类似聊天的格式。
- _可选地_，您可以传递`resize_to_multiple_of`来将嵌入层调整为`resize_to_multiple_of`参数的倍数，例如`64`。如果您希望在未来看到更多格式的支持，请在[trl](https://github.com/huggingface/trl)上打开GitHub问题

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import clone_chat_template

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 设置聊天格式
model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")
```

> [!WARNING]
> 一些基础模型，如来自Qwen的模型，在模型的分词器中有预定义的聊天模板。在这些情况下，不需要应用[`clone_chat_template()`]，因为分词器已经处理了格式化。但是，需要将EOS标记与聊天模板对齐，以确保模型的响应正确终止。在这些情况下，在[`SFTConfig`]中指定`eos_token`；例如，对于`Qwen/Qwen2.5-1.5B`，应该设置`eos_token="<|im_end|>"`。

设置好模型和分词器后，我们现在可以在对话数据集上微调我们的模型。以下是数据集如何格式化为微调的示例。

### 数据集格式支持

[`SFTTrainer`]支持流行的数据集格式。这允许您直接将数据集传递给训练器，无需任何预处理。支持以下格式：
* 对话格式
```json
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "..."}]}
```
* 指令格式
```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```

如果您的数据集使用上述格式之一，您可以直接将其传递给训练器，无需预处理。[`SFTTrainer`]将使用模型分词器中定义的格式，通过[apply_chat_template](https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models)方法为您格式化数据集。

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

...

# 加载jsonl数据集
dataset = load_dataset("json", data_files="path/to/dataset.jsonl", split="train")
# 从HuggingFace Hub加载数据集
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

...

training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    args=training_args,
    train_dataset=dataset,
)
```

如果数据集不是这些格式之一，您可以预处理数据集以匹配格式化，或者向SFTTrainer传递格式化函数来为您完成。让我们看看。

### 格式化您的输入提示

对于指令微调，在数据集中有两个列是很常见的：一个用于提示，另一个用于响应。
这允许人们像[Stanford-Alpaca](https://github.com/tatsu-lab/stanford_alpaca)那样格式化示例：
```bash
Below is an instruction ...

### Instruction
{prompt}

### Response:
{completion}
```
假设您的数据集有两个字段，`question`和`answer`。因此您可以运行：
```python
...
def formatting_prompts_func(example):
    return f"### Question: {example['question']}\n ### Answer: {example['answer']}"


trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompt_func,
)

trainer.train()
```
要正确格式化您的输入，请确保通过循环遍历所有示例并返回处理后的文本列表来处理所有示例。查看如何在alpaca数据集上使用SFTTrainer的完整示例[这里](https://github.com/huggingface/trl/pull/444#issue-1760952763)

### 打包数据集

[`SFTTrainer`]支持_示例打包_，其中多个短示例打包在同一个输入序列中以提高训练效率。要启用此数据集类的使用，只需将`packing=True`传递给[`SFTConfig`]构造函数。

```python
...
training_args = SFTConfig(packing=True)

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args
)

trainer.train()
```

请注意，如果您使用打包数据集并且如果您在训练参数中传递`max_steps`，您可能会训练模型超过几个epoch，具体取决于您配置打包数据集和训练协议的方式。请仔细检查您知道并理解您在做什么。
如果您不想打包您的`eval_dataset`，您可以在`SFTConfig`初始化方法中传递`eval_packing=False`。

#### 使用打包数据集自定义您的提示

如果您的数据集有几个您想要组合的字段，例如，如果数据集有`question`和`answer`字段并且您想要组合它们，您可以向训练器传递一个格式化函数来处理这个问题。例如：

```python
def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text

training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func
)

trainer.train()
```

### 对预训练模型的控制

您可以直接将`from_pretrained()`方法的kwargs传递给[`SFTConfig`]。例如，如果您想以不同精度加载模型，类似于

```python
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.bfloat16)

...

training_args = SFTConfig(
    model_init_kwargs={
        "torch_dtype": "bfloat16",
    },
    output_dir="/tmp",
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```
请注意，支持`from_pretrained()`的所有关键字参数。

### 训练适配器

我们还支持与🤗 PEFT库的紧密集成，以便任何用户都可以方便地训练适配器并在Hub上共享它们，而不是训练整个模型。

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/Capybara", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    "Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=SFTConfig(output_dir="Qwen2.5-0.5B-SFT"),
    peft_config=peft_config
)

trainer.train()
```

> [!WARNING]
> 如果聊天模板包含特殊标记如`<|im_start|>`（ChatML）或`<|eot_id|>`（Llama），嵌入层和LM头必须通过`modules_to_save`参数包含在可训练参数中。没有这个，微调的模型将产生无界或无意义的生成。如果聊天模板不包含特殊标记（例如Alpaca），那么可以忽略`modules_to_save`参数或设置为`None`。

您也可以继续训练您的`PeftModel`。为此，首先在`SFTTrainer`外部加载`PeftModel`，并直接将其传递给训练器，而不传递`peft_config`参数。

### 使用基础8位模型训练适配器

为此，您需要首先在训练器外部加载您的8位模型，并向训练器传递`PeftConfig`。例如：

```python
...

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-125m",
    load_in_8bit=True,
    device_map="auto",
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(),
    peft_config=peft_config,
)

trainer.train()
```

## 使用Flash Attention和Flash Attention 2

您可以使用SFTTrainer开箱即用地从Flash Attention 1和2中受益，只需最少的代码更改。
首先，为了确保您拥有transformers的所有最新功能，从源代码安装transformers

```bash
pip install -U git+https://github.com/huggingface/transformers.git
```

请注意，Flash Attention现在只在GPU上工作，并且在半精度模式下（当使用适配器时，基础模型以半精度加载）
还要注意，这两个功能与其他工具如量化完全兼容。

### 使用Flash-Attention 1

对于Flash Attention 1，您可以使用`BetterTransformer` API并强制调度API使用Flash Attention内核。首先，安装最新的optimum包：

```bash
pip install -U optimum
```

加载模型后，在`with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):`上下文管理器下包装`trainer.train()`调用：

```diff
...

+ with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    trainer.train()
```

请注意，您不能在任意数据集上使用Flash Attention 1训练模型，因为如果您使用Flash Attention内核，`torch.scaled_dot_product_attention`不支持使用填充标记进行训练。因此，您只能在使用`packing=True`时使用该功能。如果您的数据集包含填充标记，请考虑切换到Flash Attention 2集成。

以下是在单个NVIDIA-T4 16GB上使用Flash Attention 1在速度和内存效率方面可以获得的一些数字。

| use_flash_attn_1 | model_name        | max_seq_len | batch_size | time per training step |
| ---------------- | ----------------- | ----------- | ---------- | ---------------------- |
| ✓                | facebook/opt-350m | 2048        | 8          | ~59.1s                 |
|                  | facebook/opt-350m | 2048        | 8          | **OOM**                |
| ✓                | facebook/opt-350m | 2048        | 4          | ~30.3s                 |
|                  | facebook/opt-350m | 2048        | 4          | ~148.9s                |

### 使用Flash Attention-2

要使用Flash Attention 2，首先安装最新的`flash-attn`包：

```bash
pip install -U flash-attn
```

并在调用`from_pretrained`时添加`attn_implementation="flash_attention_2"`：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    attn_implementation="flash_attention_2"
)
```

如果您不使用量化，请确保您的模型以半精度加载并调度到支持的GPU设备上。
加载模型后，您可以按原样训练它，或者在模型被量化的情况下附加适配器并在其上训练适配器。

与Flash Attention 1相比，集成使得在包含填充标记的任意数据集上训练模型成为可能。

### 使用模型创建工具

我们包含了一个创建模型的工具函数。

[[autodoc]] ModelConfig

```python
from trl import ModelConfig, SFTTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
model_args = ModelConfig(
    model_name_or_path="facebook/opt-350m"
    attn_implementation=None, # 或 "flash_attention_2"
)
torch_dtype = (
    model_args.torch_dtype
    if model_args.torch_dtype in ["auto", None]
    else getattr(torch, model_args.torch_dtype)
)
quantization_config = get_quantization_config(model_args)
model_kwargs = dict(
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
    attn_implementation=model_args.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False if training_args.gradient_checkpointing else True,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
trainer = SFTTrainer(
    ...,
    model=model_args.model_name_or_path,
    peft_config=get_peft_config(model_args),
)
```

### 使用NEFTune增强模型性能

NEFTune是一种提升聊天模型性能的技术，由Jain等人的论文["NEFTune: Noisy Embeddings Improve Instruction Finetuning"](https://huggingface.co/papers/2310.05914)提出。它包括在训练期间向嵌入向量添加噪声。根据论文摘要：

> 使用Alpaca对LLaMA-2-7B进行标准微调在AlpacaEval上达到29.79%，使用噪声嵌入后上升到64.69%。NEFTune在现代指令数据集上也优于强基线。使用Evol-Instruct训练的模型看到10%的改进，使用ShareGPT看到8%的改进，使用OpenPlatypus看到8%的改进。即使是经过RLHF进一步改进的强大模型，如LLaMA-2-Chat，也能从NEFTune的额外训练中受益。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/neft-screenshot.png">
</div>

要在`SFTTrainer`中使用它，只需在创建`SFTConfig`实例时传递`neftune_noise_alpha`。请注意，为了避免任何意外行为，NEFTune在训练后被禁用，以恢复到嵌入层的原始行为。

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    neftune_noise_alpha=5,
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```

我们通过在[OpenAssistant数据集](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)上训练`mistralai/Mistral-7B-v0.1`来测试NEFTune，并验证使用NEFTune在MT Bench上带来了约25%的性能提升。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-neftune-mistral-7b.png">
</div>

但请注意，性能提升的数量_取决于数据集_，特别是在[UltraChat](https://huggingface.co/datasets/stingning/ultrachat)等合成数据集上应用NEFTune通常会产生较小的提升。

### 使用`unsloth`加速微调2倍

您可以使用与`SFTTrainer`完全兼容的[`unsloth`](https://github.com/unslothai/unsloth)库进一步加速QLoRA / LoRA（2倍更快，内存减少60%）。目前，`unsloth`仅支持Llama（Yi、TinyLlama、Qwen、Deepseek等）和Mistral架构。下面列出了1x A100上的一些基准：

| 1 A100 40GB     | Dataset   | 🤗   | 🤗 + Flash Attention 2 | 🦥 Unsloth | 🦥 VRAM saved |
| --------------- | --------- | --- | --------------------- | --------- | ------------ |
| Code Llama 34b  | Slim Orca | 1x  | 1.01x                 | **1.94x** | -22.7%       |
| Llama-2 7b      | Slim Orca | 1x  | 0.96x                 | **1.87x** | -39.3%       |
| Mistral 7b      | Slim Orca | 1x  | 1.17x                 | **1.88x** | -65.9%       |
| Tiny Llama 1.1b | Alpaca    | 1x  | 1.55x                 | **2.74x** | -57.8%       |

首先，根据[官方文档](https://github.com/unslothai/unsloth)安装`unsloth`。安装后，您可以以非常简单的方式将unsloth集成到您的工作流程中；您只需要加载`FastLanguageModel`而不是`AutoModelForCausalLM`，如下所示：

```python
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

max_length = 2048 # 支持自动RoPE缩放，所以选择任何数字

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_length,
    dtype=None,  # None用于自动检测。Float16用于Tesla T4、V100，Bfloat16用于Ampere+
    load_in_4bit=True,  # 使用4位量化减少内存使用。可以是False
    # token = "hf_...", # 如果使用像meta-llama/Llama-2-7b-hf这样的门控模型，请使用一个
)

# 进行模型修补并添加快速LoRA权重
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0目前被优化
    bias="none",  # Bias = "none"目前被优化
    use_gradient_checkpointing=True,
    random_state=3407,
)

training_args = SFTConfig(output_dir="./output", max_length=max_length)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

保存的模型与Hugging Face的transformers库完全兼容。在[官方仓库](https://github.com/unslothai/unsloth)中了解更多关于unsloth的信息。

## Liger-Kernel：多GPU训练提高20%吞吐量并减少60%内存

[Liger Kernel](https://github.com/linkedin/Liger-Kernel)是专门为LLM训练设计的Triton内核集合。它可以有效地将多GPU训练吞吐量提高20%，并将内存使用量减少60%。这样，我们可以**4倍**我们的上下文长度，如下面的基准所示。他们已经实现了Hugging Face兼容的`RMSNorm`、`RoPE`、`SwiGLU`、`CrossEntropy`、`FusedLinearCrossEntropy`，还有更多即将到来。内核与[Flash Attention](https://github.com/Dao-AILab/flash-attention)、[PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)和[Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)开箱即用。

通过这种内存减少，您可以潜在地关闭`cpu_offloading`或梯度检查点以进一步提高性能。

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

1. 要在[`SFTTrainer`]中使用Liger-Kernel，首先通过以下方式安装：

```bash
pip install liger-kernel
```

2. 安装后，在[`SFTConfig`]中设置`use_liger_kernel`。不需要其他更改！

```python
training_args = SFTConfig(
    use_liger_kernel=True,
    ...
)
```

要了解更多关于Liger-Kernel的信息，请访问他们的[官方仓库](https://github.com/linkedin/Liger-Kernel/)。

## 最佳实践

使用该训练器训练模型时，请注意以下最佳实践：

- [`SFTTrainer`]默认总是将序列截断到[`SFTConfig`]的`max_length`参数。如果没有传递，训练器将从分词器检索该值。一些分词器不提供默认值，所以有一个检查来检索1024和该值之间的最小值。训练前请确保检查它。
- 对于在8位训练适配器，您可能需要调整PEFT的`prepare_model_for_kbit_training`方法的参数，因此我们建议用户使用`prepare_in_int8_kwargs`字段，或在[`SFTTrainer`]外部创建`PeftModel`并传递它。
- 对于使用适配器的更内存高效训练，您可以以8位加载基础模型，为此只需在创建[`SFTTrainer`]时添加`load_in_8bit`参数，或在训练器外部以8位创建基础模型并传递它。
- 如果您在训练器外部创建模型，请确保不要向训练器传递任何与`from_pretrained()`方法相关的额外关键字参数。

## 多GPU训练

训练器（因此SFTTrainer）支持多GPU训练。如果您使用`python script.py`运行脚本，它将默认使用DP作为策略，这可能[比预期慢](https://github.com/huggingface/trl/issues/1303)。要使用DDP（通常推荐，更多信息请参见[这里](https://huggingface.co/docs/transformers/en/perf_train_gpu_many?select-gpu=Accelerate#data-parallelism)），您必须使用`python -m torch.distributed.launch script.py`或`accelerate launch script.py`启动脚本。要使DDP工作，您还必须检查以下内容：
- 如果您使用gradient_checkpointing，请在TrainingArguments中添加以下内容：`gradient_checkpointing_kwargs={'use_reentrant':False}`（更多信息请参见[这里](https://github.com/huggingface/transformers/issues/26969)
- 确保模型放置在正确的设备上：
```python
from accelerate import PartialState
device_string = PartialState().process_index
model = AutoModelForCausalLM.from_pretrained(
     ...
    device_map={'':device_string}
)
```

## GPTQ转换

完成训练后，您可能会遇到GPTQ量化的一些问题。将`gradient_accumulation_steps`降低到`4`将解决量化到GPTQ格式过程中的大多数问题。

## 扩展`SFTTrainer`以支持视觉语言模型

`SFTTrainer`本身不支持视觉语言数据。但是，我们提供了如何调整训练器以支持视觉语言数据的指南。具体来说，您需要使用与视觉语言数据兼容的自定义数据整理器。本指南概述了进行这些调整的步骤。对于具体示例，请参考脚本[`examples/scripts/sft_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py)，它演示了如何在[HuggingFaceH4/llava-instruct-mix-vsft](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)数据集上微调LLaVA 1.5模型。

### 准备数据

数据格式是灵活的，只要它与我们稍后将定义的自定义整理器兼容。常见的方法是使用对话数据。由于数据包括文本和图像，格式需要相应调整。以下是涉及文本和图像的对话数据格式示例：

```python
images = ["obama.png"]
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Who is this?"},
            {"type": "image"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Barack Obama"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is he famous for?"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "He is the 44th President of the United States."}
        ]
    }
]
```

为了说明如何使用LLaVA模型处理这种数据格式，您可以使用以下代码：

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print(processor.apply_chat_template(messages, tokenize=False))
```

输出将格式化为：

```txt
Who is this? ASSISTANT: Barack Obama USER: What is he famous for? ASSISTANT: He is the 44th President of the United States. 
```

<iframe src="https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft/embed/viewer/default/train" frameborder="0" width="100%" height="560px"></iframe>

### 处理多模态数据的自定义整理器

与`SFTTrainer`的默认行为不同，处理多模态数据是在数据整理过程中动态完成的。为此，您需要定义一个处理文本和图像的自定义整理器。此整理器必须将示例列表作为输入（有关数据格式的示例，请参见上一节），并返回一批处理后的数据。以下是此类整理器的示例：

```python
def collate_fn(examples):
    # 获取文本和图像，并应用聊天模板
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"][0] for example in examples]

    # 对文本进行分词并处理图像
    batch = processor(texts, images, return_tensors="pt", padding=True)

    # 标签是input_ids，我们在损失计算中屏蔽填充标记
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch
```

我们可以通过运行以下代码验证整理器按预期工作：

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train")
examples = [dataset[0], dataset[1]]  # 仅为了示例的两个示例
collated_data = collate_fn(examples)
print(collated_data.keys())  # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'labels'])
```

### 训练视觉语言模型

现在我们已经准备了数据并定义了整理器，我们可以继续训练模型。为了确保数据不被处理为仅文本，我们需要在`SFTConfig`中设置几个参数，特别是`remove_unused_columns`和`skip_prepare_dataset`为`True`，以避免数据集的默认处理。以下是如何设置`SFTTrainer`的示例。

```python
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    processing_class=processor.tokenizer,
)
```

在[HuggingFaceH4/llava-instruct-mix-vsft](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)数据集上训练LLaVa 1.5的完整示例可以在脚本[`examples/scripts/sft_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py)中找到。

- [实验跟踪](https://wandb.ai/huggingface/trl/runs/2b2c5l7s)
- [训练模型](https://huggingface.co/HuggingFaceH4/sft-llava-1.5-7b-hf)

## SFTTrainer

[[autodoc]] SFTTrainer

## SFTConfig

[[autodoc]] SFTConfig

## 数据集

在SFTTrainer中，我们智能地支持`datasets.IterableDataset`以及其他样式的数据集。如果您使用不想全部保存到磁盘的大型语料库，这很有用。数据将在飞行中进行分词和处理，即使启用了打包。

此外，在SFTTrainer中，如果它们是`datasets.Dataset`或`datasets.IterableDataset`，我们支持预分词数据集。换句话说，如果这样的数据集有`input_ids`列，将不会进行进一步的处理（分词或打包），数据集将按原样使用。如果您在此脚本之外预分词了数据集并想要直接重用它，这可能很有用。 