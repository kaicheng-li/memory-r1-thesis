import torch
import json
from typing import List, Dict, Any, Tuple
from ..common.registry import MODULE_REGISTRY
from .base import BaseModule
from ..common.utils import parse_json_from_text
from ..envs.memory_bank_utils import MemoryItem, generate_id

RERANK_PROMPT = """You are an intelligent memory assistant tasked with retrieving
accurate information from conversation memories.
# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant
to answering the question.
# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"),
calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words.
9. IMPORTANT: Select memories you found that are useful for answering the questions,
and output it before you answer questions.
10. IMPORTANT: Output the final answer after **Answer:**
# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions
Memories for user John:- 7:20 pm on 16 June, 2023: John has a special memory of a vacation to California where he experienced a
gorgeous sunset and an enjoyable night strolling the shore, creating meaningful memories with loved ones.- 6:13 pm on 10 April, 2023: John explored the coast in the Pacific Northwest and visited some national
parks, finding the beauty of nature absolutely breathtaking.- 3:14 pm on 13 August, 2023: John enjoys spending time outdoors with his family, including activities
such as hiking, hanging out at the park, and having picnics. He also values indoor family activities like
playing board games and having movie nights at home.
... (In total 30 most relevant memories from John's Memory Bank are provided) ...
Memories for user Maria:- 6:29 pm on 7 July, 2023: John experienced a severe flood in his old area last week, which caused
significant damage to homes due to poor infrastructure.- 1:24 pm on 25 May, 2023: Maria appreciates the beauty of small, meaningful moments in life, as reflected
in her reaction to a family beach photo shared by John.- 3:14 pm on 13 August, 2023: Maria appreciates family bonding and is interested in the activities that
John and his family enjoy doing together.
... (In total 30 most relevant memories from Maria's Memory Bank are provided) ...
Question: Does John live close to a beach or the mountains?"""


def build_answer_input(question: str, memories_by_speaker: Dict[str, List[Dict[str, Any]]]) -> str:
    """Attach runtime retrieved memories to the canonical Answer prompt."""
    sections = []
    for speaker, memories in memories_by_speaker.items():
        sections.append(
            f"Memories for user {speaker}:\n"
            f"{json.dumps(memories, ensure_ascii=False, indent=2)}\n"
            f"... (In total {len(memories)} relevant memories from {speaker}'s Memory Bank are provided) ..."
        )
    return (
        f"{RERANK_PROMPT}\n\n"
        + "\n\n".join(sections)
        + f"\n\nQuestion: {question}\n"
    )

@MODULE_REGISTRY.register("naive_retriever")
class NaiveRetriever(BaseModule):
    """
    Naive Retriever that wraps a store/search function.
    Not trainable.
    """
    def rollout(self, model, batch_data, **kwargs):
        return None

    def inference(self, batch_data, **kwargs):
        # This module doesn't generate text in the RL sense, 
        # it retrieves documents.
        pass
    
    def retrieve(self, query: str, store: Any, top_k: int = 3):
        # search similar documents in the store 
        # Required: Memstore with method "search_similar"
        # so actually we seldom use this retriever in the framework, instead we use store.search directly.
        if hasattr(store, 'search_similar'):
             results = store.search_similar(query, top_k=top_k)
             return [m for m, s in results]
        return []

@MODULE_REGISTRY.register("rerank_retriever")
class RerankRetriever(BaseModule):
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        self.max_generate_length = kwargs.get("max_generate_length", 512)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate_with_pytorch(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_generate_length,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        return outputs

    def get_id_map(self, context_memory, extraction_output):
        context_memory_objs = [ MemoryItem.from_dict(mem) for mem in context_memory ]
        id_counter = 1
        id_map = {}
        
        for mem in context_memory_objs:
            temp_id = id_counter
            id_counter += 1
            id_map[temp_id] = ("context", mem)
            
        ext_json = parse_json_from_text(extraction_output)
        if ext_json and "memory_list" in ext_json and isinstance(ext_json["memory_list"], list):
            for item in ext_json["memory_list"]:
                if not isinstance(item, dict): continue
                temp_id = id_counter
                id_counter += 1
                id_map[temp_id] = ("candidate", item)
        
        return id_map

    def rollout(self, model, batch_data,
                extraction_texts: List[str],
                update_texts: List[str],
                store: Any,
                reward_fn: Any = None,
                num_generations: int = 1
        ):
        
        if not store:
            print("Warning: Store not provided to RerankRetriever.rollout")
            return None, None, {}

        bs = len(batch_data['fact'])
        prompts = []
        candidates_list = [] # Store candidates for evaluation

        # Prepare prompts
        for i in range(bs):
            memory = batch_data['memory'][i]
            query = batch_data['query'][i]
            ctx_mem = batch_data['context_memory'][i]
            ext_out = extraction_texts[i]
            upd_out = update_texts[i]
            
            # 1. Reset Store
            if store.use_mock:
                store.from_list(memory)
            else:
                assert False, "we recommand use mock"
            
            # 2. Apply Update
            upd_json = parse_json_from_text(upd_out)
            id_map = self.get_id_map(ctx_mem, ext_out)
            
            update_err = False
            if upd_json and "operations" in upd_json:
                for op in upd_json["operations"]:
                    temp_id = op.get("id")
                    action = op.get("op", "NONE").upper()
                    if temp_id not in id_map: 
                        update_err = True; break
                    origin_type, origin_obj = id_map[temp_id]
                    
                    if origin_type == "context":
                        if action == "DEL": store.delete(origin_obj.id)
                        elif action != "NONE": update_err = True; break
                    elif origin_type == "candidate":
                        if action in ["ADD", "UPDATE"]:
                            key = op.get("key", origin_obj.get("key"))
                            value = op.get("value", origin_obj.get("value"))
                            new_mem = MemoryItem(id=generate_id(), 
                                key=key, value=value,
                                memory_type=origin_obj.get("memory_type", "UserMemory"),
                                tags=origin_obj.get("tags", [])
                            )
                            store.save(new_mem)
            else:
                update_err = True
            
            if update_err:
                # Skip retrieval if update failed
                candidates_list.append([])
                prompts.append("SKIP") # Placeholder
                continue

            # 3. Retrieve Candidates
            # Even if update failed, we try to retrieve from whatever state
            results = store.search_similar(query, top_k=30)
            candidates = [m for m, s in results]
            candidates_list.append(candidates)
            
            # 4. Construct Prompt
            candidates_str = "\n".join([f"[ID: {idx+1}] {m.key}: {m.value}" for idx, m in enumerate(candidates)])
            prompt = RERANK_PROMPT.format(query=query, candidates=candidates_str)
            prompts.append(prompt)

        # Expand prompts for num_generations
        batch_prompts = []
        # Need to handle skipped items
        valid_indices = []
        
        for idx, p in enumerate(prompts):
            if p != "SKIP":
                batch_prompts.extend([p] * num_generations)
                valid_indices.extend([idx] * num_generations)
            
        # Generate Rerank Responses
        if not batch_prompts:
            return [], [], {}
            
        msgs_list = [[{"role": "user", "content": p}] for p in batch_prompts]
        formatted_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
        
        tokenized = self.tokenizer(formatted_prompts, padding=True, return_tensors='pt').to(self.device)
        outputs = self._generate_with_pytorch(model, tokenized['input_ids'], tokenized['attention_mask'])
        
        input_len = tokenized['input_ids'].size(1)
        generated_ids = outputs[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Reconstruct full list (including skipped ones as empty/None)
        # generated_texts only corresponds to valid_indices
        full_generated_texts = [""] * (bs * num_generations)
        full_prompts = [""] * (bs * num_generations)
        
        gen_ptr = 0
        for i in range(bs):
            if prompts[i] == "SKIP":
                for j in range(num_generations):
                    global_idx = i * num_generations + j
                    full_generated_texts[global_idx] = "" # Empty generation
                    full_prompts[global_idx] = ""
            else:
                for j in range(num_generations):
                    global_idx = i * num_generations + j
                    if gen_ptr < len(generated_texts):
                        full_generated_texts[global_idx] = generated_texts[gen_ptr]
                        full_prompts[global_idx] = formatted_prompts[gen_ptr]
                        gen_ptr += 1

        # Compute Rewards
        assert reward_fn is not None, "Reward function must be provided"
        scores = {}
        if reward_fn:
            # Expand candidates list to match generated_texts size
            expanded_candidates = []
            for c in candidates_list:
                expanded_candidates.extend([c] * num_generations)
                
            predictions = {
                'rerank': full_generated_texts,
                'candidates': expanded_candidates # List[List[MemoryItem]]
            }
            
            scores = reward_fn(
                predictions=predictions,
                ground_truths=batch_data,
                num_generations=num_generations
            )
            # Filter out invalid entries before returning
            valid_prompts = []
            valid_generated_texts = []
            valid_scores = []
            
            has_scores = scores.get('rerank') is not None
            
            for i in range(len(full_prompts)):
                if full_prompts[i] != "":
                    valid_prompts.append(full_prompts[i])
                    valid_generated_texts.append(full_generated_texts[i])
                    if has_scores:
                        valid_scores.append(scores['rerank'][i].item())
            # Log rewards if swanlab is available
            try:
                import swanlab
                log_dict = {}
                if scores.get('rerank') is not None:                    
                    log_dict["train/reward_rerank_mean"] = torch.tensor(valid_scores).mean().item()
                    log_dict["train/reward_rerank_std"] = torch.tensor(valid_scores).std().item()
                if log_dict:
                    swanlab.log(log_dict)
            except ImportError:
                pass
        else:
            scores = {'rerank': None}
            
        
                    
        if has_scores:
            scores['rerank'] = torch.tensor(valid_scores, dtype=torch.float32, device=self.device)
            
        return valid_prompts, valid_generated_texts, scores

    def inference(self, batch_data, **kwargs):
        pass

    def retrieve(self, query: str, store: Any, top_k: int = 3):
        pass
