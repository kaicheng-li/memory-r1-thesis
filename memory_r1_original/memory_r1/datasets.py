from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .agents import FactExtractor
from .memory_bank import MemoryBank
from .retriever import LexicalMemoryRetriever
from .schemas import AnswerSample, DialogueTurn, ManagerSample, MemoryEntry, QuestionAnswer


def _parse_turn(row: Dict) -> DialogueTurn:
    return DialogueTurn(
        speaker=row["speaker"],
        text=row["text"],
        timestamp=row.get("timestamp", ""),
        metadata={k: v for k, v in row.items() if k not in {"speaker", "text", "timestamp"}},
    )


def _parse_qa(row: Dict) -> QuestionAnswer:
    return QuestionAnswer(
        question=row["question"],
        answer=row["answer"],
        turn_index=row.get("turn_index"),
        metadata={k: v for k, v in row.items() if k not in {"question", "answer", "turn_index"}},
    )


def build_manager_samples(dialogues: List[Dict], extractor: FactExtractor, history_window: int = 50) -> List[ManagerSample]:
    samples: List[ManagerSample] = []
    retriever = LexicalMemoryRetriever()

    for dialogue in dialogues:
        participants = dialogue["participants"]
        turns = [_parse_turn(turn) for turn in dialogue["turns"]]
        questions = [_parse_qa(qa) for qa in dialogue.get("questions", [])]
        memory_bank = MemoryBank()

        for turn_index, turn in enumerate(turns):
            facts = extractor.extract(turn)
            snapshot = memory_bank.clone()
            snapshot_entries = snapshot.entries()

            linked_questions = [qa for qa in questions if qa.turn_index == turn_index]
            for qa in linked_questions:
                samples.append(
                    ManagerSample(
                        dialogue_id=dialogue["dialogue_id"],
                        participants=participants,
                        memory_snapshot=snapshot_entries,
                        current_turn=turn,
                        question=qa.question,
                        answer=qa.answer,
                    )
                )

            for fact in facts:
                memory_bank.add(
                    speaker=turn.speaker,
                    text=fact,
                    timestamp=turn.timestamp,
                    source_turn=turn_index,
                )

            if len(memory_bank.entries()) > history_window:
                kept = memory_bank.entries()[-history_window:]
                memory_bank = MemoryBank(kept)

    return samples


def build_answer_samples(
    dialogues: List[Dict],
    extractor: FactExtractor,
    per_speaker_top_k: int = 30,
) -> List[AnswerSample]:
    samples: List[AnswerSample] = []
    retriever = LexicalMemoryRetriever()

    for dialogue in dialogues:
        participants = dialogue["participants"]
        turns = [_parse_turn(turn) for turn in dialogue["turns"]]
        questions = [_parse_qa(qa) for qa in dialogue.get("questions", [])]
        memory_bank = MemoryBank()

        for turn_index, turn in enumerate(turns):
            facts = extractor.extract(turn)
            for fact in facts:
                memory_bank.add(
                    speaker=turn.speaker,
                    text=fact,
                    timestamp=turn.timestamp,
                    source_turn=turn_index,
                )

        for qa in questions:
            retrieved = retriever.retrieve_per_speaker(
                query=qa.question,
                memories=memory_bank.entries(),
                participants=participants,
                per_speaker_top_k=per_speaker_top_k,
            )
            samples.append(
                AnswerSample(
                    dialogue_id=dialogue["dialogue_id"],
                    participants=participants,
                    question=qa.question,
                    answer=qa.answer,
                    retrieved_memories=retrieved,
                )
            )

    return samples


def manager_sample_to_row(sample: ManagerSample) -> Dict:
    return {
        "dialogue_id": sample.dialogue_id,
        "participants": sample.participants,
        "memory_snapshot": [asdict(entry) for entry in sample.memory_snapshot],
        "current_turn": asdict(sample.current_turn),
        "question": sample.question,
        "answer": sample.answer,
    }


def answer_sample_to_row(sample: AnswerSample) -> Dict:
    return {
        "dialogue_id": sample.dialogue_id,
        "participants": sample.participants,
        "question": sample.question,
        "answer": sample.answer,
        "retrieved_memories": [asdict(entry) for entry in sample.retrieved_memories],
    }
