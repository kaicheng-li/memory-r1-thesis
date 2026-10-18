"""Reports only: reference and candidate distributions never filter samples."""
import json
from pathlib import Path
from statistics import mean, median

from memfactory.reference_answer import atomic_json


def distribution(values):
    bins = [0] * 10
    for value in values:
        bins[min(int(value*10), 9)] += 1
    return {"n":len(values), "mean":mean(values) if values else None,
            "median":median(values) if values else None,
            "min":min(values) if values else None, "max":max(values) if values else None,
            "zeros":sum(v == 0 for v in values), "ones":sum(v == 1 for v in values),
            "histogram_edges":[i/10 for i in range(11)], "histogram_counts":bins}


def reference_report(cache, references, output, candidate_logs=()):
    records = []
    for entry in references.entries.values():
        record = cache.read(entry)
        if record is None:
            raise ValueError(f"Missing reference cache for {entry['dialogue_id']}:{entry['question_id']}")
        records.append(record)
    scores = {(r["dialogue_id"], r["question_id"]):r["f1"] for r in records}
    paired, candidate_values = [], []
    for path in candidate_logs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "candidate_group":
                continue
            key = (str(row["dialogue_id"]), str(row["question_id"]))
            if key not in scores:
                continue
            # Refuse statistics from a different input/QA contract.
            entry = references.entries[key]
            from memfactory.reference_answer import digest
            if row.get("input_hash") != digest(entry):
                raise ValueError(f"Candidate/reference data mismatch: {key}")
            rewards = [c["reward"] for c in row["candidates"]]
            if not rewards:
                continue
            paired.append({"key":key, "all_negative":all(v < scores[key] for v in rewards)})
            candidate_values.extend(rewards)
    report = {"protocol_hash":cache.protocol_hash, "reference":distribution(list(scores.values())),
              "candidates":distribution(candidate_values), "paired_groups":len(paired),
              "empirical_all_negative_fraction":mean(r["all_negative"] for r in paired) if paired else None,
              "input_truncated":False, "selection_policy":"report_only_no_filter",
              "comparison_note":("Prior-run paired candidates; not a prediction of future training."
                                 if paired else "No compatible per-QA candidate log: all-negative frequency is unknown.")}
    output = Path(output)
    atomic_json(output, report)
    summary = ["# 原文参考分数报告", "", "本报告仅诊断，不筛题、不决定是否允许训练。", "",
               f"参考题数：{len(records)}；平均 F1：{report['reference']['mean']:.4f}；"
               f"零分题：{report['reference']['zeros']}；满分题：{report['reference']['ones']}。", "",
               f"可配对的历史候选组：{len(paired)}。", report["comparison_note"], "",
               "| F1 区间 | 原文参考题数 | 历史候选数 |", "|---|---:|---:|"]
    for i in range(10):
        summary.append(f"| {i/10:.1f}–{(i+1)/10:.1f} | {report['reference']['histogram_counts'][i]} | {report['candidates']['histogram_counts'][i]} |")
    if paired:
        summary.extend(["", f"历史配对候选的全负组比例：{report['empirical_all_negative_fraction']:.2%}（不是未来训练保证）。"])
    output.with_suffix(".md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    return report
