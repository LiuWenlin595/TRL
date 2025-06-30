# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --dataset_streaming \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
    --report_to wandb

# LoRA:
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --dataset_streaming \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
    --report_to wandb
"""

import argparse

import torch
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def main(script_args, training_args, model_args):
    ################
    # 显存监控
    ###################
    print(f"初始显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    # model_kwargs = dict(
    #     revision=model_args.model_revision,
    #     attn_implementation=model_args.attn_implementation,
    #     torch_dtype=torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    # )
    model_dir = "D:/A_Code/model/Qwen/Qwen2-0.5B-Instruct/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,  # 使用半精度
        device_map="cpu",  # 自动设备映射  # auto
        use_cache=False,  # 禁用KV缓存以节省显存
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        # 如果使用LoRA，不需要ref_model
        ref_model = None
    else:
        ref_model = None
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    # dataset = load_dataset(
    #     script_args.dataset_name,
    #     name=script_args.dataset_config,
    #     streaming=script_args.dataset_streaming,
    # )
    dataset = load_from_disk("D:/A_Code/datasets/feedback_binarized")

    ##########
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.dataset_name = "trl-lib/ultrafeedback_binarized"
    script_args.dataset_streaming = True
    script_args.model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
    script_args.learning_rate = 5.0e-7
    script_args.num_train_epochs = 1
    script_args.per_device_train_batch_size = 1  # 2
    script_args.gradient_accumulation_steps = 4  # 8
    # 进一步减少批次大小
    script_args.per_device_train_batch_size = 1
    script_args.gradient_accumulation_steps = 2  # 总批次大小从4降到2
    script_args.gradient_checkpointing = True
    script_args.logging_steps = 25
    script_args.eval_strategy = "steps"
    script_args.eval_steps = 50
    script_args.output_dir = "Qwen2-0.5B-DPO"
    script_args.no_remove_unused_columns = True
    script_args.report_to = "wandb"
    
    # 启用LoRA以减少显存使用
    script_args.use_peft = True
    script_args.lora_r = 16
    script_args.lora_alpha = 32
    script_args.lora_dropout = 0.1
    
    # 更激进的LoRA设置
    script_args.lora_r = 8  # 减少rank
    script_args.lora_alpha = 16
    script_args.lora_dropout = 0.1
    # 只训练特定层
    script_args.lora_target_modules = ["q_proj", "v_proj"]  # 只训练注意力层
    
    # 使用8位优化器节省显存
    training_args.optim = "adamw_8bit"  # 优化器状态使用8bit (节省75%)
    training_args.bf16 = True  # 模型参数和计算使用bfloat16 (节省50%)
    # 两者可以同时使用，效果叠加，总共节省约60-70%显存
    
    # 额外的显存优化
    # training_args.dataloader_pin_memory = False  # 禁用内存固定
    # training_args.remove_unused_columns = False  # 避免数据预处理时的显存占用
    # training_args.max_grad_norm = 1.0  # 梯度裁剪
    
    # 限制序列长度以减少显存使用
    training_args.max_seq_length = 32  # 限制最大序列长度
    training_args.truncation_side = "right"  # 从右侧截断

    main(script_args, training_args, model_args)
