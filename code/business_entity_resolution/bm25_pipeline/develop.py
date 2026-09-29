"""Measure BM25 candidate recall on the frozen Source 1 development split."""

from __future__ import annotations

import argparse
from pathlib import Path

from runner import ROOT, run_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "student_resource/dataset/train")
    parser.add_argument("--artifacts-dir", type=Path,
                        default=ROOT / "code/business_entity_resolution/artifacts/v2")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "code/business_entity_resolution/artifacts/bm25_pipeline/development")
    parser.add_argument("--backend", choices=("scipy", "bm25s", "cupy"), default="bm25s")
    parser.add_argument("--docs-per-country", type=int, default=150_000)
    parser.add_argument("--targets-per-country", type=int, default=20_000)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    run_partition(
        split=1, partition_name="development", data_dir=args.train_dir,
        artifacts=args.artifacts_dir, output_dir=args.output_dir,
        backend=args.backend, docs_per_country=args.docs_per_country,
        targets_per_country=args.targets_per_country, cap=args.cap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
