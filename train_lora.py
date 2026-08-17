"""
训练LoRA模型 - 核心训练脚本
"""
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import argparse
import os

def load_data(data_path: str, max_samples: int = None):
    """加载处理好的训练数据"""
    prompts = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            prompts.append(data['prompt'])
    return prompts

def tokenize_function(examples, tokenizer, max_length=2048):
    """tokenize函数"""
    # tokenize
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    
    # 对于因果LM，labels就是input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/train_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./lora_model")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 开始训练...")
    print(f"模型: {args.model_name}")
    print(f"数据: {args.data_path}")
    
    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载数据
    texts = load_data(args.data_path, args.max_samples)
    dataset = Dataset.from_dict({"text": texts})
    
    # 3. tokenize数据
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    print(f"📊 训练样本数: {len(tokenized_dataset)}")
    
    # 4. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 5. 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",  # 不用wandb
        remove_unused_columns=False,
    )
    
    # 7. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 8. 创建Trainer并训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 9. 保存模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ 训练完成！模型保存至 {args.output_dir}")

if __name__ == "__main__":
    main()