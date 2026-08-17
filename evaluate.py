"""
评估脚本 - 测试微调后的模型效果
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from tqdm import tqdm
import argparse
import os

def load_test_data(data_path: str, n: int = 50):
    """加载测试数据"""
    test_examples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            test_examples.append(data)
    return test_examples

def generate_answer(model, tokenizer, prompt: str, max_new_tokens=100):
    """生成答案"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    # 只取新生成的部分
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def compute_metrics(predictions, references):
    """计算ROUGE和BERTScore"""
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    
    rouge_result = rouge.compute(predictions=predictions, references=references)
    bert_result = bertscore.compute(
        predictions=predictions, 
        references=references, 
        lang="zh"  # 如果数据是中文，改成"en"
    )
    
    return {
        'rouge1': rouge_result['rouge1'],
        'rouge2': rouge_result['rouge2'],
        'rougeL': rouge_result['rougeL'],
        'bertscore': sum(bert_result['f1']) / len(bert_result['f1'])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="./lora_model")
    parser.add_argument("--data_path", type=str, default="data/train_data.jsonl")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="results.json")
    args = parser.parse_args()
    
    print("🔍 开始评估...")
    
    # 1. 加载测试数据
    test_data = load_test_data(args.data_path, args.n_samples)
    print(f"📊 加载了 {len(test_data)} 个测试样本")
    
    # 2. 加载基础模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 加载LoRA模型
    print("🔄 加载LoRA模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    
    # 4. 生成答案
    predictions = []
    references = []
    
    for example in tqdm(test_data, desc="生成答案"):
        # 提取问题和完整prompt
        prompt = example['prompt'].split('回答：')[0] + '回答：'
        answer = example['answer']
        
        # 生成
        pred = generate_answer(model, tokenizer, prompt)
        
        predictions.append(pred)
        references.append(answer)
        
        # 打印几个例子看看
        if len(predictions) <= 3:
            print(f"\n📝 问题: {example['question'][:50]}...")
            print(f"✅ 正确: {answer}")
            print(f"🤖 生成: {pred}")
            print("-" * 50)
    
    # 5. 计算指标
    print("\n📈 计算指标...")
    metrics = compute_metrics(predictions, references)
    
    print("\n🎯 评估结果：")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"BERTScore: {metrics['bertscore']:.4f}")
    
    # 6. 保存结果
    results = {
        "metrics": metrics,
        "samples": [
            {"question": ex['question'], "answer": ex['answer'], "prediction": pred}
            for ex, pred in zip(test_data, predictions)
        ]
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果保存至 {args.output_file}")

if __name__ == "__main__":
    main()