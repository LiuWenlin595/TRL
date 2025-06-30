# 示例


## 介绍

这些示例应该可以在以下任何设置中工作（使用相同的脚本）：
   - 单GPU
   - 多GPU（使用PyTorch分布式模式）
   - 多GPU（使用DeepSpeed ZeRO-Offload阶段1、2和3）
   - fp16（混合精度）、fp32（正常精度）或bf16（bfloat16精度）

要在这些各种模式中运行，首先使用`accelerate config`初始化accelerate配置

**注意：要训练4位或8位模型**，请运行

```bash
pip install --upgrade trl[quantization]
```


## Accelerate配置
对于所有示例，您需要使用以下命令生成🤗 Accelerate配置文件：

```shell
accelerate config # 将提示您定义训练配置
```

然后，建议使用`accelerate launch`启动任务！


# 维护的示例

脚本可以用作如何使用TRL训练器的示例。它们位于[`trl/scripts`](https://github.com/huggingface/trl/blob/main/trl/scripts)目录中。此外，我们在[`examples/scripts`](https://github.com/huggingface/trl/blob/main/examples/scripts)目录中提供示例。这些示例定期维护和测试。

| 文件 | 描述 |
| --- | --- |
| [`examples/scripts/alignprop.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/alignprop.py) | 此脚本展示如何使用[`AlignPropTrainer`]微调扩散模型。 |
| [`examples/scripts/bco.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/bco.py) | 此脚本展示如何使用[`KTOTrainer`]和BCO损失微调模型，使用[openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)数据集来提高指令遵循、真实性、诚实性和有用性。 |
| [`examples/scripts/cpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py) | 此脚本展示如何使用[`CPOTrainer`]微调模型，使用[Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)数据集来提高有用性和无害性。 |
| [`examples/scripts/ddpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ddpo.py) | 此脚本展示如何使用[`DDPOTrainer`]使用强化学习微调稳定扩散模型。 |
| [`examples/scripts/dpo_online.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_online.py) | 此脚本展示如何使用[`OnlineDPOTrainer`]微调模型。 |
| [`examples/scripts/dpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_vlm.py) | 此脚本展示如何使用[`DPOTrainer`]微调视觉语言模型以减少幻觉，使用[openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)数据集。 |
| [`examples/scripts/gkd.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/gkd.py) | 此脚本展示如何使用[`GKDTrainer`]微调模型。 |
| [`examples/scripts/nash_md.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/nash_md.py) | 此脚本展示如何使用[`NashMDTrainer`]微调模型。 |
| [`examples/scripts/orpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) | 此脚本展示如何使用[`ORPOTrainer`]微调模型，使用[Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)数据集来提高有用性和无害性。 |
| [`examples/scripts/ppo/ppo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py) | 此脚本展示如何使用[`PPOTrainer`]微调模型，以提高其继续具有积极情感或物理描述性语言文本的能力 |
| [`examples/scripts/ppo/ppo_tldr.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py) | 此脚本展示如何使用[`PPOTrainer`]微调模型，以提高其生成TL;DR摘要的能力。 |
| [`examples/scripts/prm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/prm.py) | 此脚本展示如何使用[`PRMTrainer`]微调过程监督奖励模型（PRM）。 |
| [`examples/scripts/reward_modeling.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py) | 此脚本展示如何使用[`RewardTrainer`]在您自己的数据集上训练结果奖励模型（ORM）。 |
| [`examples/scripts/rloo/rloo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/rloo/rloo.py) | 此脚本展示如何使用[`RLOOTrainer`]微调模型，以提高其继续具有积极情感或物理描述性语言文本的能力 |
| [`examples/scripts/rloo/rloo_tldr.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/rloo/rloo_tldr.py) | 此脚本展示如何使用[`RLOOTrainer`]微调模型，以提高其生成TL;DR摘要的能力。 |
| [`examples/scripts/sft_gemma3.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_gemma3.py) | 此脚本展示如何使用[`SFTTrainer`]微调Gemma 3模型。 |
| [`examples/scripts/sft_video_llm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_video_llm.py) | 此脚本展示如何使用[`SFTTrainer`]微调视频语言模型。 |
| [`examples/scripts/sft_vlm_gemma3.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py) | 此脚本展示如何使用[`SFTTrainer`]在视觉到文本任务上微调Gemma 3模型。 |
| [`examples/scripts/sft_vlm_smol_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_smol_vlm.py) | 此脚本展示如何使用[`SFTTrainer`]微调SmolVLM模型。 |
| [`examples/scripts/sft_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py) | 此脚本展示如何在聊天设置中使用[`SFTTrainer`]微调视觉语言模型。该脚本仅在[LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf)、[LLaVA 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)和[Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)模型上进行了测试，因此用户在其他模型架构中可能会看到意外行为。 |
| [`examples/scripts/xpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/xpo.py) | 此脚本展示如何使用[`XPOTrainer`]微调模型。 |

这里还有一些更容易运行的colab笔记本，您可以用它们开始使用TRL：

| 文件 | 描述 |
| --- | --- |
| [`examples/notebooks/best_of_n.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/best_of_n.ipynb) | 此笔记本演示了在使用PPO微调模型时如何使用TRL的"Best of N"采样策略。 |
| [`examples/notebooks/gpt2-sentiment.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-sentiment.ipynb) | 此笔记本演示了如何在jupyter笔记本上重现GPT2 imdb情感调优示例。 |
| [`examples/notebooks/gpt2-control.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-control.ipynb) | 此笔记本演示了如何在jupyter笔记本上重现GPT2情感控制示例。 |


我们还有一些其他维护较少的示例，但可以用作参考：
1. **[research_projects](https://github.com/huggingface/trl/tree/main/examples/research_projects)**：查看此文件夹以找到一些使用TRL的研究项目脚本（LM去毒化、Stack-Llama等）


## 分布式训练

所有脚本都可以通过在使用`accelerate launch`时提供🤗 Accelerate配置文件的路径在多个GPU上运行。要在一个或多个GPU上启动其中一个，请运行以下命令（将`{NUM_GPUS}`替换为您机器中的GPU数量，将`--all_arguments_of_the_script`替换为您的参数。）

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

您还可以调整🤗 Accelerate配置文件的参数以满足您的需求（例如混合精度训练）。

### 使用DeepSpeed的分布式训练

大多数脚本可以与DeepSpeed ZeRO-{1,2,3}一起在多个GPU上运行，以高效地分片优化器状态、梯度和模型权重。为此，请运行以下命令（将`{NUM_GPUS}`替换为您机器中的GPU数量，将`--all_arguments_of_the_script`替换为您的参数，将`--deepspeed_config`替换为DeepSpeed配置文件的路径，例如`examples/deepspeed_configs/deepspeed_zero1.yaml`）：

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
``` 