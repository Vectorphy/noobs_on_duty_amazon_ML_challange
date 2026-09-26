"""Render the completed v2 training metrics as a compact Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def candidate_section(title: str, data: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", "", "| Measure | Result |", "|---|---:|"]
    lines.extend([
        f"| Candidate pairs | {data.get('candidate_pairs', '—'):,} |" if isinstance(data.get("candidate_pairs"), int) else f"| Candidate pairs | {data.get('candidate_pairs', '—')} |",
        f"| Mean candidates per Source 1 | {number(data.get('mean_candidates_per_s1'), 2)} |",
        f"| All true links retained | {number(data.get('all_true_links_retained_fraction'))} |",
        f"| Candidate-stage oracle macro F₀.₅ | {number(data.get('oracle_macro_f05'))} |",
        f"| Entities without candidates | {data.get('entities_without_candidates', '—')} |",
        "",
        "| Target source | True links | Retrieved | Recall |",
        "|---|---:|---:|---:|",
    ])
    for source, stats in sorted(data.get("by_source", {}).items()):
        lines.append(
            f"| Source {source} | {stats.get('links', '—'):,} | "
            f"{stats.get('retrieved', '—'):,} | {number(stats.get('recall'))} |"
            if isinstance(stats.get("links"), int) and isinstance(stats.get("retrieved"), int)
            else f"| Source {source} | {stats.get('links', '—')} | {stats.get('retrieved', '—')} | {number(stats.get('recall'))} |"
        )
    lines.append("")
    return lines


def render_report(metrics: dict[str, Any]) -> str:
    rows = metrics.get("rows", {})
    training = metrics.get("training", {})
    dev = metrics.get("final_development", {})
    validation = metrics.get("final_validation", {})
    diagnostics = metrics.get("candidate_diagnostics", {})
    baseline = metrics.get("baseline_15_features", {})
    features = metrics.get("selected_features", [])
    cap = metrics.get("selected_cap_per_source", "—")
    threshold = dev.get("threshold")

    lines = [
        "# V2 Full-Corpus Validation Report", "",
        "## Run summary", "",
        f"- Random seed: {metrics.get('seed', '—')}",
        f"- Source 1 entities: {rows.get('s1', '—'):,}" if isinstance(rows.get("s1"), int) else f"- Source 1 entities: {rows.get('s1', '—')}",
        f"- Source 2 / Source 3 records: {rows.get('s2', '—'):,} / {rows.get('s3', '—'):,}" if isinstance(rows.get("s2"), int) and isinstance(rows.get("s3"), int) else f"- Source 2 / Source 3 records: {rows.get('s2', '—')} / {rows.get('s3', '—')}",
        f"- Retrieved training positives / sampled negatives: {training.get('retrieved_train_positives', '—'):,} / {training.get('sampled_train_negatives', '—'):,}" if isinstance(training.get("retrieved_train_positives"), int) and isinstance(training.get("sampled_train_negatives"), int) else "- Retrieved training positives / sampled negatives: —",
        f"- Candidate cap: {cap} per target source",
        f"- Decision threshold selected on development data: {number(threshold, 2)}",
        f"- Selected features ({len(features)}): {', '.join(features) if features else '—'}",
        "",
        "## Candidate-generation diagnostics", "",
    ]
    for key, label in (("development", "Development"), ("validation", "Final validation")):
        if key in diagnostics:
            lines.extend(candidate_section(label, diagnostics[key]))

    lines.extend([
        "## Held-out model score", "",
        f"- Final validation macro F₀.₅: **{number(validation.get('macro_f05'))}**",
        f"- Pair precision / recall (micro diagnostics): {number(validation.get('pair_precision'))} / {number(validation.get('pair_recall'))}",
        "",
        "| Slice | Entities | Macro F₀.₅ |",
        "|---|---:|---:|",
    ])
    for family, groups in validation.get("slices", {}).items():
        for label, stats in groups.items():
            lines.append(f"| {family}: {label} | {stats.get('entities', '—')} | {number(stats.get('macro_f05'))} |")
    lines.extend([
        "",
        "## Baseline comparison", "",
        f"The existing 15-feature model scored **{number(baseline.get('development_macro_f05'))}** macro F₀.₅ on development data at threshold {number(baseline.get('development_threshold'), 2)}. It used the same training pairs and v2 candidate lists as the selected model.",
        f"The selected model scored **{number(dev.get('macro_f05'))}** on development data at threshold {number(threshold, 2)}. The final validation score above was measured once after selection.",
        "",
        "## Limits", "",
        "Training and held-out validation contain US and India. France appears only in the test set, so these validation results do not establish performance on France. Test records and labels were not used in this experiment.",
        "",
        "This is the experimental v2 model report. The existing submission inference pipeline remains unchanged.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=root / "artifacts" / "v2" / "training_metrics.json")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "v2" / "validation_report.md")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(metrics), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
