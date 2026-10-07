"""Add the Memory-R1 LLM-as-a-Judge metric to checkpoint audit outputs.

The generation audit and semantic judge intentionally run in separate
processes so the Manager/Answer models do not share GPU memory with the judge.
Existing valid judge labels are retained, making interrupted runs resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer,
which you will score as CORRECT or WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time-related questions, the gold answer will be a specific date, month, or year. The generated answer might include relative references (e.g., "last Tuesday"), but you should be generous - if it refers to the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs. "7 May").

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Return the label in JSON format with the key as "label"."""


def extract_label(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            label = str(payload.get("label", "")).upper().strip()
            if label in {"CORRECT", "WRONG"}:
                return label
    labels = re.findall(r"\b(CORRECT|WRONG)\b", text.upper())
    return labels[-1] if labels else None


class OpenAICompatibleJudge:
    def __init__(self, model: str, base_url: str | None, api_key: str, max_tokens: int):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class LocalJudge:
    def __init__(self, model_path: str, device: str, max_prompt_tokens: int, max_tokens: int):
        from evaluation.audit_checkpoints import AuditModel

        self.model = AuditModel(model_path, None, device)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        output, _, _ = self.model.generate(
            prompt, self.max_prompt_tokens, self.max_tokens, 0.0, 0
        )
        return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = records
    for record in records:
        groups[str(record.get("category_name", "unknown"))].append(record)
    return {
        name: {
            "count": len(group),
            "judged_count": sum(record.get("llm_judge") is not None for record in group),
            "f1": mean(group, "f1"),
            "bleu1": mean(group, "bleu1"),
            "llm_judge": mean(group, "llm_judge"),
        }
        for name, group in groups.items()
    }


def update_summary(summary_path: Path, records: list[dict[str, Any]], judge_info: dict[str, Any]) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = aggregate(records)
    summary["llm_judge"] = metrics["overall"]["llm_judge"]
    summary["llm_judge_count"] = metrics["overall"]["judged_count"]
    summary["llm_judge_total"] = metrics["overall"]["count"]
    summary["metric_scorer"] = "memory_r1_locomo_v1"
    summary["judge"] = judge_info
    if summary.get("component") == "manager":
        summary["paper_metrics_all"] = metrics
        complete = [record for record in records if record.get("trajectory_complete")]
        summary["paper_metrics_complete"] = aggregate(complete)
    else:
        summary["paper_metrics"] = metrics
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)


def judge_record(
    judge: Any, record: dict[str, Any], retries: int, retry_delay: float
) -> tuple[float | None, str]:
    prediction = str(record.get("prediction", "")).strip()
    if not prediction:
        return 0.0, "Empty prediction is WRONG without a judge call."
    prompt = JUDGE_PROMPT.format(
        question=record.get("question", ""),
        gold_answer=record.get("gold", ""),
        generated_answer=prediction,
    )
    last_output = ""
    for attempt in range(retries):
        try:
            last_output = judge.generate(prompt)
            label = extract_label(last_output)
            if label is not None:
                return float(label == "CORRECT"), last_output
        except Exception as exc:
            last_output = f"[judge error] {type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(retry_delay * (attempt + 1))
    return None, last_output or "[judge error] no valid label"


def score_file(path: Path, judge: Any, args: argparse.Namespace, judge_info: dict[str, Any]) -> None:
    records = read_jsonl(path)
    pending = [
        index
        for index, record in enumerate(records)
        if record.get("llm_judge") not in {0, 0.0, 1, 1.0}
        or record.get("llm_judge_config") != judge_info
    ]
    lock = threading.Lock()
    completed = 0

    def score(index: int) -> tuple[int, float | None, str]:
        value, raw = judge_record(judge, records[index], args.retries, args.retry_delay)
        return index, value, raw

    worker_count = 1 if args.backend == "local" else args.max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(score, index) for index in pending]
        for future in concurrent.futures.as_completed(futures):
            index, value, raw = future.result()
            with lock:
                records[index]["llm_judge"] = value
                records[index]["llm_judge_raw"] = raw
                records[index]["llm_judge_config"] = judge_info
                completed += 1
                if completed % args.save_every == 0 or completed == len(pending):
                    write_jsonl_atomic(path, records)
                    print(f"judge {path.name}: {completed}/{len(pending)}", flush=True)

    summary_path = path.with_suffix(".json")
    if summary_path.exists():
        update_summary(summary_path, records, judge_info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir")
    parser.add_argument("--backend", choices=["openai", "local"], default="openai")
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--local-model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--pattern", default="*.jsonl")
    args = parser.parse_args()
    if args.backend == "local" and not args.local_model_path:
        parser.error("--backend local requires --local-model-path")
    if args.backend == "openai" and not args.model:
        parser.error("--backend openai requires --model or OPENAI_JUDGE_MODEL")
    if (
        args.backend == "openai"
        and not args.base_url
        and args.api_key in {"", "EMPTY"}
    ):
        parser.error(
            "OpenAI judge requires OPENAI_API_KEY/JUDGE_API_KEY, or set "
            "OPENAI_BASE_URL/JUDGE_BASE_URL for a compatible local service"
        )
    if min(args.max_workers, args.max_tokens, args.retries, args.save_every) < 1:
        parser.error("workers, tokens, retries, and save-every must be positive")

    if args.backend == "local":
        judge = LocalJudge(
            args.local_model_path, args.device, args.max_prompt_tokens, args.max_tokens
        )
        judge_model = args.local_model_path
    else:
        judge = OpenAICompatibleJudge(args.model, args.base_url, args.api_key, args.max_tokens)
        judge_model = args.model
    judge_info = {
        "backend": args.backend,
        "model": judge_model,
        "base_url": args.base_url if args.backend == "openai" else None,
        "prompt": "memory_r1_correct_wrong_v1",
    }

    paths = sorted(Path(args.audit_dir).glob(args.pattern))
    if not paths:
        parser.error(f"no audit record files matched {args.pattern!r}")
    for path in paths:
        score_file(path, judge, args, judge_info)


if __name__ == "__main__":
    main()
