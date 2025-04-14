import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from data.dataset import get_dataset
from datetime import datetime
import json

def get_target_modules(model_path):
    # QWEN模型常见的注意力模块命名
    if "Qwen" in model_path:
        return ["q_proj", "k_proj", "v_proj"]
    else:
        raise ValueError(f"未知模型结构: {model_path}")

def run(args):
    logging.debug(f"开始 LoRA 微调: 模型={args.model_path}, 数据={args.data_path}, 轮次={args.epochs}")

    # 1) 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    logging.info("基座模型加载完成")

    # 3) LoRA 配置
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=get_target_modules(args.model_path),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    # 4) 加载并分词数据集
    dataset = get_dataset(tokenizer, args.data_path, args.model_path)
    train_dataset = dataset
    logging.debug("数据加载完成")

    # 5) 生成保存目录
    if args.output_dir:
        save_dir = args.output_dir
    else:
        # 默认目录（向后兼容）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"output/lora_{args.model_path.replace('/', '_')}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)

    # 6) 训练参数
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        # 如果没有验证集可写 "no"
        evaluation_strategy="no",
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        logging_dir=os.path.join(save_dir, "logs"),
        save_total_limit=2,
        report_to="none"
    )

    # 7) 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer
    )

    # 8) 开始训练
    trainer.train()

    # 9) 保存微调后模型
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
    
    logging.debug(f"LoRA 微调完成，模型保存在：{save_dir}")
