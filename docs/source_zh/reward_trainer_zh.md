# 奖励建模

[![](https://img.shields.io/badge/All_models-Reward_Trainer-blue)](https://huggingface.co/models?other=reward-trainer,trl)

TRL 支持自定义奖励建模，任何人都可以在自己的数据集和模型上执行奖励建模。

查看完整的灵活示例 [`examples/scripts/reward_modeling.py`](https://github.com/huggingface/trl/tree/main/examples/scripts/reward_modeling.py)。

## 预期数据集类型

[`RewardTrainer`] 需要 [*隐式提示*偏好数据集](dataset_formats#preference)。这意味着数据集应该只包含 `"chosen"` 和 `"rejected"` 列（而不是 `"prompt"`）。
[`RewardTrainer`] 支持 [对话式](dataset_formats#conversational) 和 [标准](dataset_formats#standard) 数据集格式。当提供对话式数据集时，训练器将自动对数据集应用聊天模板。

您也可以使用预标记化的数据集，在这种情况下，数据集应包含以下列：`input_ids_chosen`、`attention_mask_chosen`、`input_ids_rejected` 和 `attention_mask_rejected`。

## 使用 `RewardTrainer`

准备数据集后，您可以像使用 🤗 Transformers 中的 `Trainer` 类一样使用 [`RewardTrainer`]。
您应该将 `AutoModelForSequenceClassification` 模型传递给 [`RewardTrainer`]，同时传递配置训练超参数的 [`RewardConfig`]。

### 利用 🤗 PEFT 训练奖励模型

只需在 [`RewardTrainer`] 的关键字参数中传递 `peft_config`，训练器应该自动处理将模型转换为 PEFT 模型！

```python
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

model = AutoModelForSequenceClassification.from_pretrained("gpt2")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

...

trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

```

### 为损失函数添加边界

如 [Llama 2 论文](https://huggingface.co/papers/2307.09288) 中所述，您可以通过向数据集添加 `margin` 列为损失函数添加边界。奖励整理器将自动传递它，并相应地计算损失。

```python
def add_margin(row):
    # 假设您有 score_chosen 和 score_rejected 列，您想用它们来计算边界
    return {'margin': row['score_chosen'] - row['score_rejected']}

dataset = dataset.map(add_margin)
```

### 居中奖励

在许多场景中，最好确保奖励模型的输出均值为零。这通常通过首先计算模型的平均分数然后减去它来完成。

[[Eisenstein et al., 2023]](https://huggingface.co/papers/2312.09244) 提出了一个辅助损失函数，旨在直接学习居中的奖励模型。这个辅助损失最小化奖励的平方和，鼓励模型自然地产生均值为零的输出：

$$\Big( R(p, r_1) + R(p, r_2) \Big)^2 $$

这个辅助损失与主损失函数结合，由 `[RewardConfig]` 中的参数 `center_rewards_coefficient` 加权。默认情况下，此功能被停用（`center_rewards_coefficient = None`）。

```python
training_args = RewardConfig(
    center_rewards_coefficient=0.01,
    ...
)
```

有关参考结果，请参阅 PR [#1932](https://github.com/huggingface/trl/pull/1932)。

## RewardTrainer

[[autodoc]] RewardTrainer

## RewardConfig

[[autodoc]] RewardConfig 