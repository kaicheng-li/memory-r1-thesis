"""
准备训练数据 - 将LongMemEval转为[记忆+问题→答案]格式
"""
import json
import random
from typing import List, Dict
import os

def load_longmemeval_data(file_path: str) -> List[Dict]:
    """加载LongMemEval数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"✅ 加载了 {len(data)} 条数据")
    return data

def session_to_text(session: Dict) -> str:
    """将会话转为文本"""
    turns = []
    for turn in session.get('turns', []):
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        turns.append(f"{role}: {content}")
    return "\n".join(turns)

def get_evidence_sessions(instance: Dict, all_sessions: List[Dict]) -> List[str]:
    """获取证据会话的文本"""
    evidence_ids = instance.get('answer_session_ids', [])
    evidence_texts = []
    
    for session in all_sessions:
        if session.get('id') in evidence_ids:
            evidence_texts.append(session_to_text(session))
    
    return evidence_texts

def get_negative_sessions(instance: Dict, all_sessions: List[Dict], n: int = 2) -> List[str]:
    """获取负样本（无关会话）"""
    evidence_ids = instance.get('answer_session_ids', [])
    negative_sessions = []
    
    for session in all_sessions:
        if session.get('id') not in evidence_ids:
            negative_sessions.append(session)
    
    # 随机选n个
    selected = random.sample(negative_sessions, min(n, len(negative_sessions)))
    return [session_to_text(s) for s in selected]

def create_training_example(instance: Dict, all_sessions: List[Dict]) -> Dict:
    """创建单个训练样本"""
    # 1. 获取正例记忆（证据会话）
    pos_memories = get_evidence_sessions(instance, all_sessions)
    
    # 2. 获取负例记忆（无关会话）- 让模型学会忽略无关信息
    neg_memories = get_negative_sessions(instance, all_sessions, n=1)
    
    # 3. 合并所有记忆
    all_memories = pos_memories + neg_memories
    random.shuffle(all_memories)  # 打乱顺序
    
    # 4. 构造记忆文本
    memory_text = ""
    for i, mem in enumerate(all_memories):
        memory_text += f"[记忆 {i+1}]:\n{mem}\n\n"
    
    # 5. 构造完整输入
    question = instance['question']
    answer = instance['answer']
    
    prompt = f"""请基于提供的记忆信息回答问题。

{memory_text}
问题：{question}

回答：{answer}"""
    
    return {
        "prompt": prompt,
        "question": question,
        "answer": answer,
        "memory_count": len(all_memories)
    }

def main():
    # 配置
    input_file = "data/longmemeval/longmemeval_s.json"  # 请修改为您的文件路径
    output_file = "data/train_data.jsonl"
    os.makedirs("data", exist_ok=True)
    
    # 加载数据
    data = load_longmemeval_data(input_file)
    
    # 创建训练样本
    train_examples = []
    for instance in data:
        try:
            example = create_training_example(instance, instance.get('haystack_sessions', []))
            train_examples.append(example)
        except Exception as e:
            print(f"⚠️ 处理失败: {e}")
    
    # 保存
    with open(output_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✅ 生成了 {len(train_examples)} 条训练数据，保存至 {output_file}")
    
    # 显示一个例子
    print("\n🎯 示例数据：")
    example = train_examples[0]
    print(f"记忆数量: {example['memory_count']}")
    print(f"问题: {example['question'][:50]}...")
    print(f"答案: {example['answer']}")
    print(f"完整prompt预览:\n{example['prompt'][:200]}...")

if __name__ == "__main__":
    main()