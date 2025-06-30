# 训练后使用模型

一旦您使用 SFTTrainer、PPOTrainer 或 DPOTrainer 训练了模型，您将拥有一个可用于文本生成的微调模型。在本节中，我们将介绍加载微调模型和生成文本的过程。如果您需要运行训练模型的推理服务器，可以探索 [`text-generation-inference`](https://github.com/huggingface/text-generation-inference) 等库。

## 加载和生成

如果您完全微调了模型，即没有使用 PEFT，您可以像加载 transformers 中的任何其他语言模型一样简单地加载它。例如，在 PPO 训练期间训练的价值头不再需要，如果您使用原始 transformer 类加载模型，它将被忽略：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "kashif/stack-llama-2" #path/to/your/model/or/name/on/hub
device = "cpu" # 或者如果您有 GPU 则使用 "cuda"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

inputs = tokenizer.encode("This movie was really", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

或者您也可以使用 pipeline：

```python
from transformers import pipeline

model_name_or_path = "kashif/stack-llama-2" #path/to/your/model/or/name/on/hub
pipe = pipeline("text-generation", model=model_name_or_path)
print(pipe("This movie was really")[0]["generated_text"])
```

## 使用适配器 PEFT

```python
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "kashif/stack-llama-2" #path/to/your/model/or/name/on/hub"
adapter_model_name = "path/to/my/adapter"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
```

您也可以将适配器合并到基础模型中，这样您就可以像使用普通的 transformers 模型一样使用该模型，但是检查点文件会显著更大：

```python
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)

model = model.merge_and_unload()
model.save_pretrained("merged_adapters")
```

一旦您加载了模型并合并了适配器或在顶部单独保留它们，您就可以像上面概述的普通模型一样运行生成。 