import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig
from peft import PromptTuningConfig, get_peft_model, TaskType
from data.dataset import get_dataset
from datetime import datetime
from torch.nn import CrossEntropyLoss

def run(args):
    logging.debug(f"开始 P-Tuning v1 微调: 模型={args.model_path}, 数据={args.data_path}, 轮次={args.epochs}")

    # 确保只支持 QWEN 系列模型
    if "Qwen" not in args.model_path:
        raise ValueError("当前仅支持 QWEN 系列模型")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    # 准备 P-Tuning 配置
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,  # QWEN的虚拟token配置
        tokenizer_name_or_path=args.model_path,
        # 如果不支持 encoder_hidden_size, 就先注释
        # encoder_hidden_size=256
    )
    model = get_peft_model(model, peft_config)

    # 加载并分词数据集
    dataset = get_dataset(tokenizer, args.data_path, args.model_path)
    train_dataset = dataset  # 没有验证集就直接拿 dataset 当训练集

    # 输出目录
    if args.output_dir:
        save_dir = args.output_dir
    else:
        # 默认目录（向后兼容）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"output/ptuning_{args.model_path.replace('/', '_')}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)

    # 确保数据集包含labels字段
    def preprocess_function(examples):
        # ... 现有代码 ...
        
        # 将input_ids同时用作labels（用于自回归语言模型）
        result["labels"] = result["input_ids"].copy()
        
        return result

    # 训练参数
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        evaluation_strategy="no",  # 没有验证集，改成 no
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        logging_dir=os.path.join(save_dir, "logs"),
        save_total_limit=2,
        report_to="none",
        prediction_loss_only=True,  # 确保只关注预测损失
    )

    # 确保模型配置中启用了损失计算
    model_config = AutoConfig.from_pretrained(
        args.model_path,
        # ... 其他参数 ...
    )
    model_config.use_cache = False  # 训练时禁用缓存以确保正确计算梯度

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,       # 没有测试集就 None
        tokenizer=tokenizer
    )

    # 开始训练
    trainer.train()

    # 保存微调后模型之前，确保使用绝对路径
    absolute_base_model_path = os.path.abspath(args.model_path)
    
    # 更安全的方式处理peft_config，避免直接修改对象
    # 创建保存配置信息的JSON文件
    adapter_config_path = os.path.join(save_dir, "adapter_config.json")
    
    # 先保存模型
    model.save_pretrained(save_dir)
    
    # 然后手动修改adapter_config.json文件
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # 更新基座模型路径
        adapter_config['base_model_name_or_path'] = absolute_base_model_path
        
        # 写回文件
        with open(adapter_config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
    
    logging.debug(f"P-Tuning v1 微调完成，模型保存在：{save_dir}")

    # 可能需要自定义模型前向传递方法来明确计算损失
    class PrefixTuningModel(AutoModelForCausalLM):
        # ... 现有代码 ...
        
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            if labels is not None:
                # 计算语言模型损失
                logits = outputs.logits
                # 确保计算损失 - 通常用于自回归语言模型
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs.loss = loss
                
            return outputs
