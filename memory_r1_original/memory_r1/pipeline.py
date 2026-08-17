from __future__ import annotations

from typing import Dict, List

from .agents import AnswerAgent, FactExtractor, MemoryManagerAgent
from .memory_bank import MemoryBank
from .retriever import LexicalMemoryRetriever
from .schemas import AnswerTrace, DialogueTurn


class MemoryR1Pipeline:
    def __init__(
        self,
        extractor: FactExtractor,
        manager: MemoryManagerAgent,
        answer_agent: AnswerAgent,
        retriever: LexicalMemoryRetriever | None = None,
    ):
        self.extractor = extractor
        self.manager = manager
        self.answer_agent = answer_agent
        self.retriever = retriever or LexicalMemoryRetriever()

    def construct_memory_bank(self, turns: List[DialogueTurn], retrieval_top_k: int = 5) -> MemoryBank:
        bank = MemoryBank()
        for turn_index, turn in enumerate(turns):
            facts = self.extractor.extract(turn)
            query = " ".join(facts) if facts else turn.text
            retrieved = self.retriever.retrieve(query, bank.entries(), top_k=retrieval_top_k)
            decisions = self.manager.propose(turn, facts, [item.entry for item in retrieved], temperature=0.0)
            bank.apply(decisions, default_speaker=turn.speaker, timestamp=turn.timestamp)
        return bank

    def answer_question(
        self,
        question: str,
        memory_bank: MemoryBank,
        participants: List[str],
        per_speaker_top_k: int = 30,
        temperature: float = 0.0,
    ) -> AnswerTrace:
        retrieved = self.retriever.retrieve_per_speaker(
            query=question,
            memories=memory_bank.entries(),
            participants=participants,
            per_speaker_top_k=per_speaker_top_k,
        )
        return self.answer_agent.answer(question, retrieved, temperature=temperature)
