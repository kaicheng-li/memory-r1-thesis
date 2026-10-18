"""Standalone report/plots from cached references and per-rank candidate logs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memfactory.reference_answer import RawReferences, ReferenceCache
from memfactory.reference_reporting import reference_report


def plot_report(report, logs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    grouped = defaultdict(list)
    for path in logs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event") == "candidate_group":
                grouped[int(row["epoch"])].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = [i/10+.05 for i in range(10)]
    axes[0,0].bar(x, report["reference"]["histogram_counts"], width=.08, label="Raw reference QA")
    axes[0,0].set(title="Reference F1 distribution", xlabel="F1", ylabel="QA count")
    epochs = sorted(grouped)
    neg, invalid, lengths, entropies = [], [], [], []
    for epoch in epochs:
        rows = grouped[epoch]
        candidates = [c for row in rows for c in row["candidates"]]
        neg.append(sum(row["all_negative"] for row in rows)/len(rows))
        invalid.append(sum(not c["valid"] for c in candidates)/len(candidates))
        lengths.append(sum(c["completion_tokens"] for c in candidates)/len(candidates))
        measured = [c["policy_entropy"] for c in candidates if c.get("policy_entropy") is not None]
        entropies.append(sum(measured)/len(measured) if measured else float("nan"))
    axes[0,1].plot(epochs,neg,"o-",label="All-negative reference residual groups")
    axes[0,1].plot(epochs,invalid,"o-",label="Invalid actions")
    axes[0,1].set(xlabel="Epoch",ylabel="Fraction",ylim=(0,1));axes[0,1].legend(fontsize=8)
    axes[1,0].plot(epochs,lengths,"o-");axes[1,0].set(xlabel="Epoch",ylabel="Mean completion tokens")
    axes[1,1].plot(epochs,entropies,"o-");axes[1,1].set(xlabel="Epoch",ylabel="Policy entropy (nats)",title="Measured update actions only")
    if not epochs:
        for ax in (axes[0,1],axes[1,0],axes[1,1]):
            ax.text(.5,.5,"No candidate logs yet",transform=ax.transAxes,ha="center")
    fig.tight_layout();fig.savefig(Path(output).with_suffix(".svg"));plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True,help="Actual protocol.json inside the cache hash directory")
    parser.add_argument("--states",default="data/stage0/manager_states_train.jsonl")
    parser.add_argument("--timelines",default="data/stage0/dialogue_timelines_train.json")
    parser.add_argument("--candidate-log",action="append",default=[])
    parser.add_argument("--output",required=True)
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    states=[json.loads(line) for line in Path(args.states).read_text(encoding="utf-8").splitlines() if line.strip()]
    refs=RawReferences(json.loads(Path(args.timelines).read_text(encoding="utf-8")),states)
    path=Path(args.protocol)
    cache=ReferenceCache(path.parent.parent,json.loads(path.read_text(encoding="utf-8")))
    if cache.directory.resolve()!=path.parent.resolve():
        raise ValueError("Protocol hash directory mismatch")
    report=reference_report(cache,refs,args.output,args.candidate_log)
    if args.plot:plot_report(report,args.candidate_log,args.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
