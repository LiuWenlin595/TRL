# 评判器

<Tip warning={true}>

TRL评判器是一个实验性API，可能会随时发生变化。

</Tip>

TRL提供评判器来轻松比较两个完成结果。

请确保通过运行以下命令安装所需的依赖项：

```bash
pip install trl[judges]
```

## 使用提供的评判器

TRL提供了几个开箱即用的评判器。例如，您可以使用`HfPairwiseJudge`来使用Hugging Face模型中心的预训练模型比较两个完成结果：

```python
from trl import HfPairwiseJudge

judge = HfPairwiseJudge()
judge.judge(
    prompts=["法国的首都是什么？", "太阳系中最大的行星是什么？"],
    completions=[["巴黎", "里昂"], ["土星", "木星"]],
)  # 输出: [0, 1]
```

## 定义您自己的评判器

要定义您自己的评判器，我们提供了几个您可以继承的基类。对于基于排名的评判器，您需要继承[`BaseRankJudge`]并实现[`BaseRankJudge.judge`]方法。对于成对评判器，您需要继承[`BasePairJudge`]并实现[`BasePairJudge.judge`]方法。如果您想定义一个不属于这些类别的评判器，您需要继承[`BaseJudge`]并实现[`BaseJudge.judge`]方法。

作为示例，让我们定义一个偏好较短完成结果的成对评判器：

```python
from trl import BasePairwiseJudge

class PrefersShorterJudge(BasePairwiseJudge):
    def judge(self, prompts, completions, shuffle_order=False):
        return [0 if len(completion[0]) > len(completion[1]) else 1 for completion in completions]
```

然后您可以按如下方式使用这个评判器：

```python
judge = PrefersShorterJudge()
judge.judge(
    prompts=["法国的首都是什么？", "太阳系中最大的行星是什么？"],
    completions=[["巴黎", "法国的首都是巴黎。"], ["木星是太阳系中最大的行星。", "木星"]],
)  # 输出: [0, 1]
```

## 提供的评判器

### PairRMJudge

[[autodoc]] PairRMJudge

### HfPairwiseJudge

[[autodoc]] HfPairwiseJudge

### OpenAIPairwiseJudge

[[autodoc]] OpenAIPairwiseJudge

### AllTrueJudge

[[autodoc]] AllTrueJudge

## 基类

### BaseJudge

[[autodoc]] BaseJudge

### BaseBinaryJudge

[[autodoc]] BaseBinaryJudge

### BaseRankJudge

[[autodoc]] BaseRankJudge

### BasePairwiseJudge

[[autodoc]] BasePairwiseJudge 