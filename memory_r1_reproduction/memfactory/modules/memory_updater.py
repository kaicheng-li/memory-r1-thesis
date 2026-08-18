import torch
import json
from typing import Dict, Any, List, Optional, Tuple
from ..envs.memory_bank_utils import MemoryItem
from ..common.registry import MODULE_REGISTRY
from ..common.utils import parse_json_from_text
from .base import BaseModule, Samples

UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the
memory, (3) delete from the memory, and (4) no change.
Based on the above four operations, the memory will change.
Compare newly retrieved facts with the existing memory. For each new fact,
decide whether to:- ADD: Add it to the memory as a new element- UPDATE: Update an existing memory element- DELETE: Delete an existing memory element- NONE: Make no change (if the fact is already present or irrelevant)
1. **Add**: If the retrieved facts contain new information not present
in the memory, then you have to add it by generating a new ID in the id field.- Example:
Old Memory:
[
{"id" : "0", "text" : "User is a software engineer"}
]
Retrieved facts: ["Name is John"]
New Memory:
{
"memory" : [
{"id" : "0", "text" : "User is a software engineer", "event" : "NONE"},
{"id" : "1", "text" : "Name is John", "event" : "ADD"}
]
}
2. **Update**: If the retrieved facts contain information that is already
present in the memory but the information is totally different, then
you have to update it.
If the retrieved fact contains information that conveys the same thing as
the memory, keep the version with more detail.
Example (a)– if the memory contains "User likes to play cricket" and the
retrieved fact is "Loves to play cricket with friends", then update the
memory with the retrieved fact.
Example (b)– if the memory contains "Likes cheese pizza" and the
retrieved fact is "Loves cheese pizza", then do NOT update it because they
convey the same information.
Important: When updating, keep the same ID and preserve old_memory.- Example:
Old Memory:
[
{"id" : "0", "text" : "I really like cheese pizza"},
{"id" : "2", "text" : "User likes to play cricket"}
]
Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
New Memory:
{
"memory" : [
{"id" : "0", "text" : "Loves cheese and chicken pizza", "event" : "UPDATE",
"old_memory" : "I really like cheese pizza"},
{"id" : "2", "text" : "Loves to play cricket with friends", "event" : "UPDATE",
"old_memory" : "User likes to play cricket"}
]
}
3. **Delete**: If the retrieved facts contain information that contradicts
the memory, delete it. When deleting, return the same IDs — do not generate new IDs.- Example:
Old Memory:
[
{"id" : "1", "text" : "Loves cheese pizza"}
]
Retrieved facts: ["Dislikes cheese pizza"]
New Memory:
{
"memory" : [
{"id" : "1", "text" : "Loves cheese pizza", "event" : "DELETE"}
]
}
4. **No Change**: If the retrieved facts are already present, make no change.- Example:
Old Memory:
[
{"id" : "0", "text" : "Name is John"}
]
Retrieved facts: ["Name is John"]
New Memory:
{
"memory" : [
{"id" : "0", "text" : "Name is John", "event" : "NONE"}
]
}"""


def build_manager_input(old_memory: List[Dict[str, Any]], retrieved_facts: List[Any]) -> str:
    """Attach runtime memory and facts to the canonical Manager prompt."""
    return (
        f"{UPDATE_MEMORY_PROMPT}\n\n"
        "Old Memory:\n"
        f"{json.dumps(old_memory, ensure_ascii=False, indent=2)}\n\n"
        "Retrieved facts:\n"
        f"{json.dumps(retrieved_facts, ensure_ascii=False, indent=2)}\n\n"
        "Return only the JSON object with the key \"memory\"."
    )

@MODULE_REGISTRY.register("naive_updater")
class NaiveUpdater(BaseModule):
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        self.max_prompt_length = kwargs.get("max_prompt_length", 3072)
        self.max_generate_length = kwargs.get("max_generate_length", 2048)
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

    def rollout(self, model, batch_data: Dict[str, Any], extraction_texts: List[str], **kwargs) -> tuple[List[str], List[str], Dict[str, Any]]:
        """
        Rollout the update process:
        1. Receive extraction texts.
        2. Generate update plans.
        3. Call reward function to evaluate both extraction and update.
        
        Returns:
            update_prompts: List[str]
            update_responses: List[str]
            scores: Dict containing rewards
        """
        context_memories = batch_data['context_memory']
        # extraction_texts is [BS * NumGen]
        
        # 1. Generate Update Plans
        # We reuse generate logic but return prompts too
        prompts = []
        num_generations = len(extraction_texts) // len(context_memories)
        
        for i, ctx_mem in enumerate(context_memories):
            for j in range(num_generations):
                global_idx = i * num_generations + j
                ext_out = extraction_texts[global_idx]
                
                ctx_fmt, cand_fmt, id_map = self.prepare_memory_lists(ctx_mem, ext_out)
                
                prompt = UPDATE_MEMORY_PROMPT.format(
                    context_memory=json.dumps(ctx_fmt, ensure_ascii=False, indent=2),
                    candidate_memory=json.dumps(cand_fmt, ensure_ascii=False, indent=2)
                )
                prompts.append(prompt)
        
        msgs_list = [[{"role": "user", "content": p}] for p in prompts]
        formatted_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
        
        tokenized = self.tokenizer(formatted_prompts, padding=True, return_tensors='pt').to(self.device)
        outputs = self._generate_with_pytorch(model, tokenized['input_ids'], tokenized['attention_mask'])
        
        input_len = tokenized['input_ids'].size(1)
        generated_ids = outputs[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 2. Compute Rewards
        reward_fn = kwargs.get('reward_fn')
        if reward_fn:
            scores = reward_fn(
                predictions={'extraction': extraction_texts, 'update': generated_texts}, 
                ground_truths=batch_data,
                num_generations=num_generations
            )
            # Log rewards if swanlab is available
            try:
                import swanlab
                log_dict = {}
                if scores.get('extraction') is not None:
                    ext_r = scores['extraction'].float()
                    log_dict["train/reward_extraction_mean"] = ext_r.mean().item()
                    log_dict["train/reward_extraction_std"] = ext_r.std().item()
                if scores.get('update') is not None:
                    upd_r = scores['update'].float()
                    log_dict["train/reward_update_mean"] = upd_r.mean().item()
                    log_dict["train/reward_update_std"] = upd_r.std().item()
                if log_dict:
                    swanlab.log(log_dict)
            except ImportError:
                pass
        else:
            scores = {'extraction': None, 'update': None}
            
        return formatted_prompts, generated_texts, scores

    def prepare_memory_lists(self, context_memory, extraction_output):
        """
        Prepares numbered lists for prompt and mapping for execution.
        Returns:
            context_list_fmt: List[Dict] with 'id', 'key', 'value'
            candidate_list_fmt: List[Dict] with 'id', 'key', 'value'
            id_map: Dict[int, Any] - maps temp ID to (type, original_obj)
                type 'context': original_obj is MemoryItem
                type 'candidate': original_obj is Dict (from extraction)
        """
        context_memory = [ MemoryItem.from_dict(mem) for mem in context_memory ]

        id_counter = 1
        id_map = {}
        
        # 1. Process Context Memory
        context_list_fmt = []
        for mem in context_memory:
            temp_id = id_counter
            id_counter += 1
            id_map[temp_id] = ("context", mem)
            context_list_fmt.append({
                "id": temp_id,
                "key": mem.key,
                "value": mem.value
            })
            
        # 2. Process Extraction Output
        candidate_list_fmt = []
        ext_json = parse_json_from_text(extraction_output)
        if ext_json and "memory_list" in ext_json and isinstance(ext_json["memory_list"], list):
            for item in ext_json["memory_list"]:
                if not isinstance(item, dict): continue
                temp_id = id_counter
                id_counter += 1
                id_map[temp_id] = ("candidate", item)
                candidate_list_fmt.append({
                    "id": temp_id,
                    "key": item.get("key", "Unknown"),
                    "value": item.get("value", "")
                })
        
        return context_list_fmt, candidate_list_fmt, id_map


    def generate(self, model, context_memories: List[List[Any]], extraction_outputs: List[str]) -> Tuple[List[str], Samples]:
        """
        Generate update samples.
        Note: extraction_outputs is flattened [BS * NumGen]
        context_memories is [BS] -> needs to be repeated
        """
        # update.generate is Deprecated in naive algorithm
        print("Warning: memory_updater.generate is deprecated. Please use rollout instead.")
        return None

    def inference(self, llm_client, batch_data: Dict[str, Any], extraction_texts: List[str], num_generations: int = 1, **kwargs):
        assert num_generations == 1, "we recommend num_generations=1"
        # it's ok because bs=1 most of the time
        context_memories = batch_data['context_memory']
        # extraction_texts is [BS] since num_gen=1
        
        prompts = []
        for i, ctx_mem in enumerate(context_memories):
            ext_out = extraction_texts[i]
            
            ctx_fmt, cand_fmt, id_map = self.prepare_memory_lists(ctx_mem, ext_out)
            
            prompt = UPDATE_MEMORY_PROMPT.format(
                context_memory=json.dumps(ctx_fmt, ensure_ascii=False, indent=2),
                candidate_memory=json.dumps(cand_fmt, ensure_ascii=False, indent=2)
            )
            prompts.append(prompt)
            
        generated_texts = []
        for prompt in prompts:
            response = llm_client.chat("You are a smart memory manager.", prompt)
            generated_texts.append(response)
            
        return generated_texts
