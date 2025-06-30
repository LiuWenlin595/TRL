# DPO 训练器

[![](https://img.shields.io/badge/All_models-DPO-blue)](https://huggingface.co/models?other=dpo,trl) [![](https://img.shields.io/badge/smol_course-Chapter_2-yellow)](https://github.com/huggingface/smol-course/tree/main/2_preference_alignment)

## 概述

TRL 支持 DPO 训练器，用于从偏好数据训练语言模型，如 [Rafael Rafailov](https://huggingface.co/rmrafailov)、Archit Sharma、Eric Mitchell、[Stefano Ermon](https://huggingface.co/ermonste)、[Christopher D. Manning](https://huggingface.co/manning)、[Chelsea Finn](https://huggingface.co/cbfinn) 在论文 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290) 中所述。

论文摘要如下：

> 虽然大规模无监督语言模型（LM）学习广泛的世界知识和一些推理技能，但由于其训练的完全无监督性质，实现对其行为的精确控制是困难的。获得这种可操控性的现有方法收集模型生成相对质量的人工标签，并微调无监督 LM 以与这些偏好保持一致，通常通过人类反馈强化学习（RLHF）。然而，RLHF 是一个复杂且通常不稳定的过程，首先拟合反映人类偏好的奖励模型，然后使用强化学习微调大型无监督 LM 以最大化这个估计的奖励，而不会偏离原始模型太远。在本文中，我们引入了 RLHF 中奖励模型的新参数化，它能够以封闭形式提取相应的最优策略，允许我们仅使用简单的分类损失来解决标准 RLHF 问题。我们称之为直接偏好优化（DPO）的算法是稳定的、高性能的且计算轻量级的，消除了在微调期间从 LM 采样或执行重要超参数调优的需要。我们的实验表明，DPO 可以微调 LM 以与人类偏好保持一致，效果与现有方法相当或更好。值得注意的是，使用 DPO 微调在控制生成情感的能力方面超过了基于 PPO 的 RLHF，在摘要和单轮对话中匹配或改进了响应质量，同时实现和训练起来要简单得多。

【important】
第一步是训练 SFT 模型，以确保我们训练的数据对 DPO 算法来说是分布内的。

然后，通过 DPO 微调语言模型包括两个步骤，比 [PPO](ppo_trainer) 更容易：

1. **数据收集**：收集 [偏好数据集](dataset_formats#preference)，包含给定提示的正面和负面选择的生成对。
2. **优化**：直接最大化 DPO 损失的对数似然。

这个过程在下面的示意图中说明（来自 [DPO 论文的图 1](https://huggingface.co/papers/2305.18290)）：

![](https://github.com/huggingface/trl/assets/49240599/9150fac6-3d88-4ca2-8ec6-2a6f3473216d)

在 [原始论文](https://huggingface.co/papers/2305.18290) 中了解更多关于 DPO 算法的信息。

## 快速开始

这个示例演示了如何使用 DPO 方法训练模型。我们使用 [Qwen 0.5B 模型](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 作为基础模型。我们使用来自 [UltraFeedback 数据集](https://huggingface.co/datasets/openbmb/UltraFeedback) 的偏好数据。您可以在这里查看数据集中的数据：

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

以下是训练模型的脚本：

```python
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_dpo.py
```

在 8 个 GPU 上分布式训练大约需要 3 分钟。您可以通过检查奖励图来验证训练进度。奖励边界的上升趋势表明模型正在改进，并随着时间的推移生成更好的响应。

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/dpo-qwen2-reward-margin.png)

要查看 [训练好的模型](https://huggingface.co/trl-lib/Qwen2-0.5B-DPO) 的表现，您可以使用 [Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models)。

<pre><code>$ transformers chat trl-lib/Qwen2-0.5B-DPO
<strong><span style="color: red;">&lt;shirin_yamani&gt;:</span></strong>
What is Huggingface?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-DPO&gt;:</span></strong>
Huggingface is a platform that allows users to access a variety of open-source machine learning resources such as pre-trained models and datasets Huggingface is a platform that allows users to access a variety of open-source machine learning resources such as pre-trained models and datasets for the development of machine learning models and applications. It provides a repository of over 300, 000 pre-trained models in  Huggingface is a platform that allows users to access a variety of open-source machine learning resources such as pre-trained models and datasets for the development of machine learning models and applications. It provides a repository of over 300, 000  pre-trained models in a variety of languages, enabling users to explore and utilize the latest techniques and technologies in the field of machine learning.
</code></pre>

## 预期数据集类型

DPO 需要 [偏好数据集](dataset_formats#preference)。[`DPOTrainer`] 支持 [对话式](dataset_formats#conversational) 和 [标准](dataset_formats#standard) 数据集格式。当提供对话式数据集时，训练器将自动对数据集应用聊天模板。

虽然 [`DPOTrainer`] 支持显式和隐式提示，但我们建议使用显式提示。如果提供隐式提示数据集，训练器将自动从 `"chosen"` 和 `"rejected"` 列中提取提示。有关更多信息，请参阅 [偏好样式](dataset_formats#preference) 部分。

### 视觉语言模型的特殊考虑

[`DPOTrainer`] 支持微调视觉语言模型（VLM）。对于这些模型，需要视觉数据集。要了解视觉数据集的具体格式，请参阅 [视觉数据集格式](dataset_formats#vision-datasets) 部分。

此外，与使用 `tokenizer` 的标准基于文本的模型不同，对于 VLM，您应该用 `processor` 替换 `tokenizer`。

```diff
- model = AutoModelForCausalLM.from_pretrained(model_id)
+ model = AutoModelForVision2Seq.from_pretrained(model_id)

- tokenizer = AutoTokenizer.from_pretrained(model_id)
+ processor = AutoProcessor.from_pretrained(model_id)

  trainer = DPOTrainer(
      model,
      args=training_args,
      train_dataset=train_dataset,
-     processing_class=tokenizer,
+     processing_class=processor,
)
```

有关微调视觉语言模型的完整示例，请参阅 [`examples/scripts/dpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_vlm.py) 中的脚本。

## 示例脚本

我们提供了一个使用 DPO 方法训练模型的示例脚本。脚本在 [`trl/scripts/dpo.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py) 中可用

要在 [UltraFeedback 数据集](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) 上使用 [Qwen2 0.5B 模型](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 测试 DPO 脚本，请运行以下命令：

```bash
accelerate launch trl/scripts/dpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-DPO
```

## 记录的指标

在训练和评估期间，我们记录以下奖励指标：

- `rewards/chosen`：策略模型和参考模型对所选响应的对数概率差异的平均值，按 beta 缩放
- `rewards/rejected`：策略模型和参考模型对被拒绝响应的对数概率差异的平均值，按 beta 缩放
- `rewards/accuracies`：所选奖励大于相应被拒绝奖励的平均频率
- `rewards/margins`：所选奖励与相应被拒绝奖励之间的平均差异

## 损失函数

DPO 算法支持多种损失函数。损失函数可以使用 [`DPOConfig`] 中的 `loss_type` 参数设置。支持以下损失函数：

| `loss_type=`                           | 描述                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"sigmoid"` (默认)                  | 给定偏好数据，我们可以根据 Bradley-Terry 模型拟合二元分类器，实际上 [DPO](https://huggingface.co/papers/2305.18290) 作者提出了通过 `logsigmoid` 对归一化似然使用 sigmoid 损失来拟合逻辑回归。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `"hinge"`                              | [RSO](https://huggingface.co/papers/2309.06657) 作者提出在 [SLiC](https://huggingface.co/papers/2305.10425) 论文的归一化似然上使用铰链损失。在这种情况下，`beta` 是边界的倒数。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `"ipo"`                                | [IPO](https://huggingface.co/papers/2310.12036) 作者提供了对 DPO 算法的更深层次理论理解，识别了过拟合问题并提出了替代损失。在这种情况下，`beta` 是所选与被拒绝完成对的对数似然比之间差距的倒数，因此 `beta` 越小，这个差距越大。根据论文，损失在完成的对数似然上平均（与仅求和的 DPO 不同）。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `"exo_pair"`                           | [EXO](https://huggingface.co/papers/2402.00856) 作者提出最小化反向 KL 而不是 DPO 的负对数 sigmoid 损失（对应于前向 KL）。设置非零 `label_smoothing`（默认 `1e-3`）会导致成对偏好上 EXO 的简化版本（参见 [EXO 论文](https://huggingface.co/papers/2402.00856) 的等式 (16)）。EXO 的完整版本使用 SFT 策略生成的 `K>2` 个完成，当 `K` 足够大时，这成为 PPO 目标的无偏估计器（直到常数）。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `"nca_pair"`                           | [NCA](https://huggingface.co/papers/2402.05369) 作者表明 NCA 优化每个响应的绝对似然而不是相对似然。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `"robust"`                             | [Robust DPO](https://huggingface.co/papers/2403.00409) 作者提出了对数据中偏好噪声具有鲁棒性的 DPO 损失的无偏估计。与 cDPO 一样，它假设偏好标签以某种概率有噪声。在这种方法中，[`DPOConfig`] 中的 `label_smoothing` 参数用于建模现有标签噪声的概率。要应用这种保守损失，请将 `label_smoothing` 设置为大于 0.0 的值（在 0.0 和 0.5 之间；默认值为 0.0）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `"bco_pair"`                           | [BCO](https://huggingface.co/papers/2404.04656) 作者训练一个二元分类器，其 logit 作为奖励，使得分类器将 {提示，所选完成} 对映射到 1，将 {提示，被拒绝完成} 对映射到 0。对于未配对数据，我们推荐专用的 [`BCOTrainer`]。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `"sppo_hard"`                          | [SPPO](https://huggingface.co/papers/2405.00675) 作者声称 SPPO 能够通过将所选奖励推至最大 1/2 和被拒绝奖励推至最小 -1/2 来迭代求解纳什均衡，并可以缓解数据稀疏问题。该实现通过使用硬标签概率来近似这个算法，将 1 分配给获胜者，将 0 分配给失败者。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `"aot"` 或 `loss_type="aot_pair"`     | [AOT](https://huggingface.co/papers/2406.05882) 作者提出使用通过最优传输的分布偏好对齐。传统上，对齐算法在样本级别使用配对偏好，这不确保分布级别的对齐。另一方面，AOT 可以通过使正样本的奖励分布在负样本分布上在一阶上随机占优来对齐配对或未配对偏好数据的 LLM。具体来说，`loss_type="aot"` 适用于配对数据集，其中每个提示都有所选和被拒绝的响应；`loss_type="aot_pair"` 适用于未配对数据集。简而言之，`loss_type="aot"` 确保对齐模型所选与被拒绝的对数似然比具有比参考模型该比率更高的分位数。`loss_type="aot_pair"` 确保所选奖励在所有分位数上都高于被拒绝奖励。注意，在这两种情况下，分位数都是通过排序获得的。为了充分利用 AOT 算法的优势，最大化每 GPU 批次大小很重要。 |
| `"apo_zero"` 或 `loss_type="apo_down"` | [APO](https://huggingface.co/papers/2408.06266) 方法引入了对齐目标的"锚定"版本。有两个变体：`apo_zero` 和 `apo_down`。`apo_zero` 损失增加获胜输出的似然，同时减少失败输出的似然，当模型性能不如获胜输出时适用。另一方面，`apo_down` 减少获胜和失败输出的似然，但更强调减少失败输出的似然。当模型比获胜输出更好时，这个变体更有效。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `"discopop"`                           | [DiscoPOP](https://huggingface.co/papers/2406.08414) 论文使用 LLM 发现更高效的离线偏好优化损失。在论文中，提出的 DiscoPOP 损失（这是一个对数比率调制损失）在不同任务（IMDb 正面文本生成、Reddit TLDR 摘要和 Alpaca Eval 2.0）上优于其他优化损失。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

### 标签平滑

[cDPO](https://ericmitchell.ai/cdpo.pdf) 是 DPO 损失的一个调整，我们假设偏好标签以某种概率有噪声。在这种方法中，[`DPOConfig`] 中的 `label_smoothing` 参数用于建模现有标签噪声的概率。要应用这种保守损失，请将 `label_smoothing` 设置为大于 0.0 的值（在 0.0 和 0.5 之间；默认值为 0.0）。

### 同步参考模型

[TR-DPO](https://huggingface.co/papers/2404.09656) 论文建议在 DPO 训练期间，在每 `ref_model_sync_steps` 步 SGD 后使用权重 `ref_model_mixup_alpha` 同步参考模型权重。要在 [`DPOConfig`] 中切换此回调，请使用 `sync_ref_model=True`。

### RPO 损失

[RPO](https://huggingface.co/papers/2404.19733) 论文使用与本文 [paper](https://huggingface.co/papers/2405.16436) 中 RPO 损失相关的损失实现迭代偏好调优算法，该算法本质上包括对所选偏好的加权 SFT 损失以及 DPO 损失。要使用此损失，请在 [`DPOConfig`] 中将 `rpo_alpha` 设置为适当的值。论文建议将此权重设置为 `1.0`。

### WPO 损失

[WPO](https://huggingface.co/papers/2406.11827) 论文通过根据当前策略下的概率重新加权偏好对，使离策略数据更接近在策略数据。要使用此方法，请在 [`DPOConfig`] 中将 `use_weighting` 标志设置为 `True`。

### LD-DPO 损失

[LD-DPO](https://huggingface.co/papers/2409.06411) 论文基于混合系数 \\( \alpha \\) 将超过期望长度的响应部分分解为两个组件——类人偏好和冗长偏好。要使用此方法，请在 [`DPOConfig`] 中将 `ld_alpha` 设置为适当的值。论文建议将此值设置在 `0.0` 和 `1.0` 之间。

### 对于专家混合模型：启用辅助损失

如果负载在专家之间大致平均分配，MOE 是最有效的。  
为了确保我们在偏好调优期间类似地训练 MOE，将负载平衡器的辅助损失添加到最终损失是有益的。

此选项通过在模型配置中设置 `output_router_logits=True`（例如 [`~transformers.MixtralConfig`]）来启用。  
要缩放辅助损失对总损失的贡献程度，请在模型配置中使用超参数 `router_aux_loss_coef=...`（默认值：`0.001`）。

## 使用 `unsloth` 加速 DPO 微调

您可以使用与 `SFTTrainer` 完全兼容的 [`unsloth`](https://github.com/unslothai/unsloth) 库进一步加速 QLoRA / LoRA（2倍更快，内存减少 60%）。目前 `unsloth` 仅支持 Llama（Yi、TinyLlama、Qwen、Deepseek 等）和 Mistral 架构。下面列出了一些 DPO 的基准测试：

| GPU      | 模型     | 数据集    | 🤗   | 🤗 + Flash Attention 2 | 🦥 Unsloth | 🦥 VRAM 节省 |
| -------- | --------- | ---------- | --- | --------------------- | --------- | ------------ |
| A100 40G | Zephyr 7b | Ultra Chat | 1x  | 1.24x                 | **1.88x** | -11.6%       |
| Tesla T4 | Zephyr 7b | Ultra Chat | 1x  | 1.09x                 | **1.55x** | -18.6%       |

首先根据 [官方文档](https://github.com/unslothai/unsloth) 安装 `unsloth`。安装后，您可以以非常简单的方式将 unsloth 集成到您的工作流程中；您只需要加载 `FastLanguageModel` 而不是 `AutoModelForCausalLM`，如下所示：

```diff
  from datasets import load_dataset
  from trl import DPOConfig, DPOTrainer
- from transformers import AutoModelForCausalLM, AutoTokenizer
+ from unsloth import FastLanguageModel

- model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
- tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
+ model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
+ model = FastLanguageModel.get_peft_model(model)
  train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

- training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
+ training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10, bf16=True)
  trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
  trainer.train()

```

保存的模型与 Hugging Face 的 transformers 库完全兼容。在 [官方仓库](https://github.com/unslothai/unsloth) 中了解更多关于 unsloth 的信息。

## 使用 PEFT 的参考模型考虑

对于参考模型在使用 PEFT 时如何工作，您有三个主要选项（加上几个变体），假设您想要用 DPO 进一步增强的模型是使用 (Q)LoRA 调优的。

1. 简单地创建模型的两个实例，每个都加载您的适配器 - 工作正常但效率很低。
2. 将适配器合并到基础模型中，在顶部创建另一个适配器，然后将 `ref_model` 参数留空，在这种情况下 DPOTrainer 将卸载适配器进行参考推理 - 高效，但下面讨论了潜在的缺点。
3. 用不同的名称加载适配器两次，然后在训练期间使用 `set_adapter` 在正在 DPO 的适配器和参考适配器之间交换 - 与选项 2 相比效率稍低（~适配器大小 VRAM 开销），但避免了陷阱。

### 在 DPO 之前合并 QLoRA 的缺点（选项 2）

正如 [Benjamin Marie](https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997) 所建议的，合并 QLoRA 适配器的最佳选择是首先反量化基础模型，然后合并适配器。类似于 [这个脚本](https://github.com/jondurbin/qlora/blob/main/qmerge.py)。

但是，使用这种方法后，您将有一个未量化的基础模型。因此，要对 DPO 使用 QLoRA，您需要重新量化合并的模型或使用未量化的合并（导致更高的内存需求）。

### 使用选项 3 - 加载适配器两次

为了避免选项 2 的缺点，您可以将微调的适配器加载到模型中两次，使用不同的名称，并在 [`DPOTrainer`] 中设置模型/参考适配器名称。

例如：

```python
# 加载基础模型。
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mixtral-8x7b-v0.1",
    load_in_4bit=True,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

# 加载适配器。
model = PeftModel.from_pretrained(
    model,
    "/path/to/peft",
    is_trainable=True,
    adapter_name="train",
)
# 用不同的名称第二次加载适配器，这将是我们的参考模型。
model.load_adapter("/path/to/peft", adapter_name="reference")

# 初始化训练器，没有 ref_model 参数。
training_args = DPOConfig(
    model_adapter_name="train",
    ref_adapter_name="reference",
)
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    ...
)
```

## DPOTrainer

[[autodoc]] DPOTrainer

## DPOConfig

[[autodoc]] DPOConfig

## DataCollatorForPreference

[[autodoc]] trainer.dpo_trainer.DataCollatorForPreference 