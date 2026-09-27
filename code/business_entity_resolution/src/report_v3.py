"""Render the completed v3 training metrics as a compact Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def number(value: Any, digits: int = 4) -> str:
    return "\u2014" if value is None else f"{value:.{digits}f}"


def candidate_section(title: str, data: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", "", "| Measure | Result |", "|---|---:|"]
    cp = data.get("candidate_pairs")
    lines.append(
        f"| Candidate pairs | {cp:,} |" if isinstance(cp, int)
        else f"| Candidate pairs | {cp} |"
    )
    lines.extend([
        f"| Mean candidates per Source 1 | {number(data.get('mean_candidates_per_s1'), 2)} |",
        f"| All true links retained | {number(data.get('all_true_links_retained_fraction'))} |",
        f"| Candidate-stage oracle macro F\u2080.\u2085 | {number(data.get('oracle_macro_f05'))} |",
        f"| Entities without candidates | {data.get('entities_without_candidates', '\u2014')} |",
        "",
        "| Target source | True links | Retrieved | Recall |",
        "|---|---:|---:|---:|",
    ])
    for source, stats in sorted(data.get("by_source", {}).items()):
        lk = stats.get("links", "\u2014")
        rt = stats.get("retrieved", "\u2014")
        lines.append(
            f"| Source {source} | {lk:,} | {rt:,} | {number(stats.get('recall'))} |"
            if isinstance(lk, int) and isinstance(rt, int)
            else f"| Source {source} | {lk} | {rt} | {number(stats.get('recall'))} |"
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
    cap = metrics.get("selected_cap_per_source", "\u2014")
    deep_cap = metrics.get("deep_cap_per_source", "\u2014")
    deep_thresh = metrics.get("deep_candidate_threshold", "\u2014")
    threshold = dev.get("threshold")
    country_thresholds: dict = metrics.get("country_thresholds", {})
    mutual_ex: bool = metrics.get("mutual_exclusivity_suppression", False)

    s1 = rows.get("s1")
    s2 = rows.get("s2")
    s3 = rows.get("s3")
    tp = training.get("retrieved_train_positives")
    tn = training.get("sampled_train_negatives")

    lines = [
        "# V3 Full-Corpus Validation Report", "",
        "## Run summary", "",
        f"- Random seed: {metrics.get('seed', '\u2014')}",
        f"- Source 1 entities: {s1:,}" if isinstance(s1, int) else f"- Source 1 entities: {s1}",
        (f"- Source 2 / Source 3 records: {s2:,} / {s3:,}"
         if isinstance(s2, int) and isinstance(s3, int)
         else f"- Source 2 / Source 3 records: {s2} / {s3}"),
        (f"- Retrieved training positives / sampled negatives: {tp:,} / {tn:,}"
         if isinstance(tp, int) and isinstance(tn, int)
         else "- Retrieved training positives / sampled negatives: \u2014"),
        f"- Normal candidate cap: {cap} per target source",
        f"- Deep candidate cap (A3): {deep_cap} per target source"
         f" (triggered when top JW < {deep_thresh})",
        f"- Global decision threshold (from development data): {number(threshold, 2)}",
        f"- Mutual-exclusivity suppression (C2): {'enabled' if mutual_ex else 'disabled'}",
        f"- Selected features ({len(features)}): {', '.join(features) if features else '\u2014'}",
        "",
        "## Per-country thresholds (C1)", "",
    ]
    if country_thresholds:
        lines += ["| Country | Threshold | Dev F\u2080.\u2085 | N |", "|---|---:|---:|---:|"]
        for country, info in sorted(country_thresholds.items()):
            if isinstance(info, dict):
                lines.append(
                    f"| {country} | {number(info.get('threshold'), 2)} "
                    f"| {number(info.get('macro_f05'))} | {info.get('n', '\u2014')} |"
                )
            else:
                lines.append(f"| {country} | {number(info, 2)} | \u2014 | \u2014 |")
        lines.append("")
    else:
        lines += ["*No per-country thresholds recorded.*", ""]

    lines += ["## Candidate-generation diagnostics", ""]
    for key, label in (("development", "Development"), ("validation", "Final validation")):
        if key in diagnostics:
            lines.extend(candidate_section(label, diagnostics[key]))

    lines.extend([
        "## Held-out model score", "",
        f"- Final validation macro F\u2080.\u2085: **{number(validation.get('macro_f05'))}**",
        f"- Pair precision / recall: {number(validation.get('pair_precision'))} / {number(validation.get('pair_recall'))}",
        "",
        "| Slice | Entities | Macro F\u2080.\u2085 |",
        "|---|---:|---:|",
    ])
    for family, groups in validation.get("slices", {}).items():
        for label, stats in groups.items():
            lines.append(
                f"| {family}: {label} | {stats.get('entities', '\u2014')} "
                f"| {number(stats.get('macro_f05'))} |"
            )
    lines.extend([
        "",
        "## Baseline comparison", "",
        f"The 15-feature baseline scored **{number(baseline.get('development_macro_f05'))}** "
        f"macro F\u2080.\u2085 on development data at threshold "
        f"{number(baseline.get('development_threshold'), 2)}, using the same training pairs.",
        f"The selected v3 model scored **{number(dev.get('macro_f05'))}** on development data "
        f"at threshold {number(threshold, 2)}.  "
        f"The final validation score was measured once after selection.",
        "",
        "## Limits", "",
        "Training and held-out validation contain US and India. "
        "France appears only in the test set, so these validation results do not establish "
        "performance on France. Test records and labels were not used in this experiment.",
        "",
        "This is the experimental v3 model report.  "
        "The existing v2 submission inference pipeline remains unchanged.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=root / "artifacts" / "v3" / "training_metrics.json")
    parser.add_argument("--output", type=Path,
                        default=root / "artifacts" / "v3" / "validation_report.md")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(metrics), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
