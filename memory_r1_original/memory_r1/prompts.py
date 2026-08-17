from __future__ import annotations

import json
from typing import Iterable, List

from .schemas import DialogueTurn, MemoryEntry


MEMORY_MANAGER_SYSTEM_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: ADD, UPDATE, DELETE, and NONE.
Compare newly retrieved facts with the existing memory.
For each fact, decide whether to:
- ADD: Add a new memory element
- UPDATE: Update an existing memory element
- DELETE: Delete a contradictory memory element
- NONE: Keep memory unchanged

Rules:
1. Preserve IDs on UPDATE and DELETE.
2. When updating, keep the more detailed version.
3. Use NONE if the fact is already present or irrelevant.
4. Return valid JSON with key "memory".
"""


ANSWER_AGENT_SYSTEM_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

Instructions:
1. Carefully analyze all provided memories from both speakers.
2. Pay attention to timestamps when temporal reasoning matters.
3. If memories conflict, prefer the most recent relevant memory.
4. The answer should be less than 5-6 words when possible.
5. First output the IDs of useful memories.
6. Then output the final answer after the key "answer".
7. Return valid JSON with keys "selected_memory_ids" and "answer".
"""


FACT_EXTRACTION_PROMPT = """Extract durable facts from the dialogue turn.
Only keep information worth storing in long-term memory.
Return JSON with key "facts" and a list of short fact strings.
"""


def format_memory_entries(memories: Iterable[MemoryEntry]) -> str:
    payload = []
    for memory in memories:
        payload.append(
            {
                "id": memory.id,
                "speaker": memory.speaker,
                "text": memory.text,
                "timestamp": memory.timestamp,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_fact_extraction_prompt(turn: DialogueTurn) -> str:
    return (
        f"{FACT_EXTRACTION_PROMPT}\n\n"
        f"Speaker: {turn.speaker}\n"
        f"Timestamp: {turn.timestamp}\n"
        f"Turn: {turn.text}\n"
    )


def build_manager_prompt(turn: DialogueTurn, facts: List[str], retrieved_memories: List[MemoryEntry]) -> str:
    return (
        f"{MEMORY_MANAGER_SYSTEM_PROMPT}\n\n"
        f"Current speaker: {turn.speaker}\n"
        f"Current timestamp: {turn.timestamp}\n"
        f"Current turn: {turn.text}\n\n"
        f"Retrieved facts:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        f"Old Memory:\n{format_memory_entries(retrieved_memories)}\n"
    )


def build_answer_prompt(question: str, retrieved_memories: List[MemoryEntry]) -> str:
    return (
        f"{ANSWER_AGENT_SYSTEM_PROMPT}\n\n"
        f"Memories:\n{format_memory_entries(retrieved_memories)}\n\n"
        f"Question: {question}\n"
    )
