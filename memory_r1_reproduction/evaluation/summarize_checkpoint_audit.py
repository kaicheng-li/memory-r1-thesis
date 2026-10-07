"""Print a compact table from audit_checkpoints.py JSON summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_number(value, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def format_percent(value) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def print_paper_metrics(reports) -> None:
    category_order = {
        "single_hop": 0,
        "multi_hop": 1,
        "open_domain": 2,
        "temporal": 3,
        "overall": 4,
        "unknown": 5,
    }
    rows = []
    for report in reports:
        if report["component"] == "answer":
            scopes = (("all", report.get("paper_metrics", {})),)
        else:
            scopes = (
                ("all", report.get("paper_metrics_all", {})),
                ("complete", report.get("paper_metrics_complete", {})),
            )
        for scope, groups in scopes:
            for category, metrics in groups.items():
                rows.append((report.get("label", ""), scope, category, metrics))
    if not rows:
        return
    print("PAPER_METRICS_PERCENT")
    print("label\tscope\tcategory\tn\tF1\tB1\tJ\tJ_n")
    for label, scope, category, metrics in sorted(
        rows, key=lambda row: (row[0], row[1], category_order.get(row[2], 99))
    ):
        print(
            "\t".join(
                [
                    str(label),
                    scope,
                    category,
                    str(metrics.get("count", 0)),
                    format_percent(metrics.get("f1")),
                    format_percent(metrics.get("bleu1")),
                    format_percent(metrics.get("llm_judge")),
                    f"{metrics.get('judged_count', 0)}/{metrics.get('count', 0)}",
                ]
            )
        )


def print_latency(reports) -> None:
    rows = []
    for report in reports:
        for component, metrics in report.get("latency", {}).items():
            rows.append(
                (
                    report.get("label", ""),
                    component,
                    metrics,
                )
            )
    if not rows:
        return
    print("LATENCY_SECONDS")
    print("label\tcomponent\tn\tmean\tp50\tp95")
    for label, component, metrics in rows:
        print(
            "\t".join(
                [
                    str(label),
                    component,
                    str(metrics.get("count", 0)),
                    format_number(metrics.get("mean_s")),
                    format_number(metrics.get("p50_s")),
                    format_number(metrics.get("p95_s")),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir")
    args = parser.parse_args()

    reports = []
    for path in sorted(Path(args.audit_dir).glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("component") in {"manager", "answer"}:
            reports.append(report)

    answer_reports = [report for report in reports if report["component"] == "answer"]
    manager_reports = [report for report in reports if report["component"] == "manager"]
    if answer_reports:
        print("ANSWER")
        print("label\trows\tgenerations\tEM\tF1\tB1\tJ\tJ_n\tnonzero\tsignal_groups\thit_limit")
        for report in answer_reports:
            judge = report.get("llm_judge")
            print(
                "\t".join(
                    [
                        str(report.get("label", "")),
                        str(report.get("selected_rows", 0)),
                        str(report.get("generations", 0)),
                        format_number(report.get("em")),
                        format_number(report.get("f1")),
                        format_number(report.get("bleu1")),
                        format_number(judge),
                        f"{report.get('llm_judge_count', 0)}/{report.get('llm_judge_total', report.get('generations', 0))}",
                        str(report.get("nonzero_rewards", 0)),
                        f"{report.get('reward_signal_groups', 0)}/{report.get('reward_groups', 0)}",
                        str(report.get("hit_token_limit", 0)),
                    ]
                )
            )

    if manager_reports:
        print("MANAGER")
        print(
            "label\tQA\ttrajectories\tjson_valid\tcomplete\tEM_all\tF1_all\tB1_all\tJ_all\tJ_n\t"
            "nonzero\tsignal_groups\tEM_complete\tF1_complete\tB1_complete\thit_limit\tactions"
        )
        for report in manager_reports:
            judge = report.get("llm_judge")
            print(
                "\t".join(
                    [
                        str(report.get("label", "")),
                        str(report.get("selected_QA", 0)),
                        str(report.get("trajectories", 0)),
                        format_number(report.get("json_valid_rate")),
                        format_number(report.get("complete_trajectory_rate")),
                        format_number(report.get("em_all")),
                        format_number(report.get("f1_all")),
                        format_number(report.get("bleu1_all")),
                        format_number(judge),
                        f"{report.get('llm_judge_count', 0)}/{report.get('llm_judge_total', report.get('trajectories', 0))}",
                        str(report.get("nonzero_rewards", 0)),
                        f"{report.get('reward_signal_groups', 0)}/{report.get('reward_groups', 0)}",
                        format_number(report.get("em_complete")),
                        format_number(report.get("f1_complete")),
                        format_number(report.get("bleu1_complete")),
                        str(report.get("hit_token_limit_actions", 0)),
                        f"{report.get('attempted_actions', 0)}/{report.get('expected_actions', 0)}",
                    ]
                )
            )

    print_paper_metrics(reports)
    print_latency(reports)


if __name__ == "__main__":
    main()
