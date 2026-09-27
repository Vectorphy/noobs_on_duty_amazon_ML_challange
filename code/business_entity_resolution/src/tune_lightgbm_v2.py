"""Tune a small LightGBM grid on cached v2 features; leave final validation untouched."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from matching_v2 import FEATURES, FEATURE_ENGINE_VERSION
from models_v2 import fit_model
from train_v2 import (
    EXPECTED_ROWS, SEED, connect, ensure_train_selection, evaluate_model,
    extract_features, file_signature, score_grid, split_arrays,
)
from matching_v2 import f05


TRIALS = {
    "leaves_127": {"num_leaves": 127},
    "depth8_leaves63": {"num_leaves": 63, "max_depth": 8},
    "depth10_leaves127": {"num_leaves": 127, "max_depth": 10},
    "min_child500_l2_1_leaves63": {
        "num_leaves": 63, "min_child_samples": 500, "reg_lambda": 1.0,
    },
    "lr03_600_leaves63": {
        "num_leaves": 63, "learning_rate": 0.03, "n_estimators": 600,
    },
    "max_bin127_leaves63": {"num_leaves": 63, "max_bin": 127},
    "bagging8_leaves63": {
        "num_leaves": 63, "subsample": 0.8, "subsample_freq": 1,
    },
}


def source_threshold_search(prob, meta, split, cap: int, start_threshold: float) -> dict:
    """Tune separate S2/S3 thresholds on the same per-entity macro metric."""
    n = len(split["truth"])
    total = np.zeros((2, n, 101), dtype=np.int32)
    true = np.zeros_like(total)
    for start in range(0, len(meta), 50_000):
        stop = min(len(meta), start + 50_000)
        piece = meta[start:stop]
        use = piece["rank"] <= cap
        local = split["ix_to_local"][piece["s1"][use]]
        src = (piece["source"][use].astype(np.int16) - 2).astype(np.int8)
        bins = np.minimum(100, (prob[start:stop][use] * 100).astype(np.int16))
        np.add.at(total, (src, local, bins), 1)
        is_true = piece["positive"][use] == 1
        np.add.at(true, (src[is_true], local[is_true], bins[is_true]), 1)

    predicted = np.cumsum(total[:, :, ::-1], axis=2)[:, :, ::-1]
    hits = np.cumsum(true[:, :, ::-1], axis=2)[:, :, ::-1]
    steps = [int(round(start_threshold * 100))] * 2

    def score(left: int, right: int) -> float:
        pred = predicted[0, :, left] + predicted[1, :, right]
        tp = hits[0, :, left] + hits[1, :, right]
        return float(f05(split["truth"], pred, tp).mean())

    # Coordinate search keeps the exact per-entity metric while avoiding a
    # costly 101 x 101 grid. It starts at the best shared threshold.
    for _ in range(3):
        for source in range(2):
            scored = []
            for step in range(101):
                candidate = [*steps]
                candidate[source] = step
                scored.append((score(*candidate), step))
            _, steps[source] = max(scored)
    return {"macro_f05": score(*steps),
            "thresholds": {"source2": steps[0] / 100, "source3": steps[1] / 100}}


def run(artifacts: Path, sample_rows: int) -> None:
    work = artifacts / "work"
    tuning = artifacts / "tuning" / "lightgbm"
    checkpoints = tuning / "checkpoints"
    tensorboard = tuning / "tensorboard"
    tuning.mkdir(parents=True, exist_ok=True)

    required = [work / name for name in (
        "development_features.npy", "development_meta.npy", "pipeline.duckdb",
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required cached v2 files are missing: " + ", ".join(missing))

    cap = 24
    con = connect(work)
    try:
        ensure_train_selection(con, work)
        root = Path(__file__).resolve().parents[3]
        source_signature = file_signature([
            *(root / "student_resource/dataset/train" / name for name in (
                "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                "train_ground_truth.tsv",
            )),
            Path(__file__).with_name("train_v2.py"),
            Path(__file__).with_name("matching_v2.py"),
        ])
        x_train, train_meta = extract_features(
            con, work, "train", "train_selected", input_signature=source_signature,
        )
        x_dev = np.load(work / "development_features.npy", mmap_mode="r")
        dev_meta = np.load(work / "development_meta.npy", mmap_mode="r")
        if x_train.shape[1] != len(FEATURES) or x_dev.shape[1] != len(FEATURES):
            raise ValueError("Cached feature order does not match the current feature schema")
        eligible = np.flatnonzero(train_meta["rank"] <= cap)
        rng = np.random.default_rng(SEED)
        if len(eligible) > sample_rows:
            eligible = rng.choice(eligible, sample_rows, replace=False)
        sample_x = np.asarray(x_train[eligible], dtype=np.float32)
        sample_y = np.asarray(train_meta["positive"][eligible], dtype=np.uint8)
        if sample_y.min() == sample_y.max():
            raise ValueError("Tuning sample must contain both positive and negative pairs")
        dev = split_arrays(con, 1)
        input_signature = file_signature([
            *(root / "student_resource/dataset/train" / name for name in (
                "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                "train_ground_truth.tsv",
            )),
            work / "train_features.npy", work / "train_meta.npy",
            Path(__file__).with_name("train_v2.py"),
            Path(__file__).with_name("matching_v2.py"),
        ])
    finally:
        con.close()

    # Keep full entity rows together so the macro metric remains per-entity.
    eval_entities = min(30_000, len(dev["ix"]))
    chosen_positions = np.sort(rng.choice(len(dev["ix"]), eval_entities, replace=False))
    chosen_ids = dev["ix"][chosen_positions]
    dev_row_ix = np.flatnonzero(np.isin(dev_meta["s1"], chosen_ids))
    x_dev_sample = np.asarray(x_dev[dev_row_ix], dtype=np.float32)
    meta_dev_sample = np.asarray(dev_meta[dev_row_ix])
    dev_labels_sample = np.asarray(meta_dev_sample["positive"], dtype=np.uint8)
    sample_to_local = np.full(EXPECTED_ROWS["s1"], -1, dtype=np.int32)
    sample_to_local[chosen_ids] = np.arange(eval_entities, dtype=np.int32)
    dev_sample = {"truth": dev["truth"][chosen_positions],
                  "ix_to_local": sample_to_local}
    eval_root = tuning / f"evaluation_{time.time_ns()}"

    def eval_work(name: str) -> Path:
        path = eval_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    rows = []
    trial_rows = []
    best_score = -1.0
    best_model_path = tuning / "best_sample_model_rank_uniform_advanced.joblib"
    params_base = dict(
        boosting_type="gbdt", n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=100, max_bin=63,
        n_jobs=os.cpu_count() or 1, random_state=SEED,
        importance_type="gain", verbosity=-1,
    )

    current_path = artifacts / "matcher_model.joblib"
    if current_path.is_file():
        current = joblib.load(current_path)
        current_result = evaluate_model(
            current, x_dev_sample, meta_dev_sample, dev_sample, (cap,),
            tuple(range(len(FEATURES))), eval_work("current_sample"),
        )[cap]
        current_prob = np.load(eval_work("current_sample") / "current_prob.npy", mmap_mode="r")
        current_source = source_threshold_search(
            current_prob, meta_dev_sample, dev_sample, cap, current_result["threshold"],
        )
        rows.append({"trial": "current_full_model", "sample_rows": len(sample_y),
                     "macro_f05": current_result["macro_f05"],
                     "threshold": current_result["threshold"],
                     "source_threshold_result": current_source,
                     "params": current.get_params()})
        print(f"Current saved model: dev F0.5={current_result['macro_f05']:.6f}; "
              f"threshold={current_result['threshold']:.2f}", flush=True)
        del current

    for index, (name, overrides) in enumerate(TRIALS.items(), 1):
        params = {**params_base, **overrides}
        model = LGBMClassifier(**params)
        signature = {
            "model": "lightgbm", "stage": name, "params": params,
            "rows": len(sample_y), "columns": list(range(len(FEATURES))),
            "seed": SEED, "cap": cap, "feature_engine": FEATURE_ENGINE_VERSION,
            "early_stopping_rounds": 50,
            "development_rows": len(dev_labels_sample),
            "input_signature": input_signature,
        }
        started = time.monotonic()
        model = fit_model(
            "lightgbm", model, sample_x, sample_y, stage=name,
            checkpoint_dir=checkpoints, signature=signature,
            tensorboard_dir=tensorboard,
            validation_data=(x_dev_sample, dev_labels_sample),
            early_stopping_rounds=50,
        )
        result = evaluate_model(
            model, x_dev_sample, meta_dev_sample, dev_sample, (cap,),
            tuple(range(len(FEATURES))), eval_work(name),
        )[cap]
        sample_prob = np.load(eval_work(name) / "current_prob.npy", mmap_mode="r")
        source_result = source_threshold_search(
            sample_prob, meta_dev_sample, dev_sample, cap, result["threshold"],
        )
        row = {"trial": name, "sample_rows": len(sample_y),
               "macro_f05": result["macro_f05"], "threshold": result["threshold"],
               "source_threshold_result": source_result,
               "elapsed_seconds": round(time.monotonic() - started, 2), "params": params}
        rows.append(row)
        trial_rows.append(row)
        print(f"Trial {index}/{len(TRIALS)} {name}: dev F0.5={row['macro_f05']:.6f}; "
              f"threshold={row['threshold']:.2f}; {row['elapsed_seconds']:.1f}s", flush=True)
        if row["macro_f05"] > best_score:
            best_score = row["macro_f05"]
            tmp = best_model_path.with_suffix(".tmp")
            joblib.dump(model, tmp)
            os.replace(tmp, best_model_path)

    # Recheck the sample winner against every development entity before deciding
    # whether a full-data refit is justified.
    best_trial = max(trial_rows, key=lambda row: row["macro_f05"])["trial"]
    sample_blends = []
    if current_path.is_file() and best_model_path.is_file():
        p_current = np.load(eval_work("current_sample") / "current_prob.npy", mmap_mode="r")
        p_best = np.load(eval_work(best_trial) / "current_prob.npy", mmap_mode="r")
        blend_root = eval_root / "sample_blends"
        blend_root.mkdir(parents=True, exist_ok=True)
        for weight in (0.25, 0.5, 0.75):
            blend_path = blend_root / f"current_weight_{weight:.2f}.npy"
            blended = np.lib.format.open_memmap(blend_path, mode="w+", dtype="float32",
                                                shape=(len(p_current),))
            for start in range(0, len(p_current), 100_000):
                stop = min(len(p_current), start + 100_000)
                blended[start:stop] = weight * p_current[start:stop] + (1-weight) * p_best[start:stop]
            blended.flush()
            value, threshold, _ = score_grid(meta_dev_sample, blended, dev_sample, cap)
            source_result = source_threshold_search(
                blended, meta_dev_sample, dev_sample, cap, threshold,
            )
            sample_blends.append({"current_model_weight": weight,
                                  "macro_f05": value, "threshold": threshold,
                                  "source_threshold_result": source_result})
            del blended
            blend_path.unlink(missing_ok=True)

    finalists = []
    finalist_probability_paths = {}
    for name, path in (("current_full_model", current_path),
                       (best_trial, best_model_path)):
        if not path.is_file():
            continue
        model = joblib.load(path)
        result = evaluate_model(
            model, x_dev, dev_meta, dev, (cap,), tuple(range(len(FEATURES))),
            eval_work(f"full_{name}"),
        )[cap]
        finalist_probability_paths[name] = eval_work(f"full_{name}") / "current_prob.npy"
        finalists.append({"trial": name, "macro_f05": result["macro_f05"],
                          "threshold": result["threshold"]})
        print(f"Full development {name}: F0.5={result['macro_f05']:.6f}; "
              f"threshold={result['threshold']:.2f}", flush=True)

    best_sample_blend = max(sample_blends, key=lambda row: row["macro_f05"]) if sample_blends else None
    if best_sample_blend and all(name in finalist_probability_paths
                                 for name in ("current_full_model", best_trial)):
        weight = best_sample_blend["current_model_weight"]
        p_current = np.load(finalist_probability_paths["current_full_model"], mmap_mode="r")
        p_best = np.load(finalist_probability_paths[best_trial], mmap_mode="r")
        blend_path = eval_root / "full_blended_probabilities.npy"
        blended = np.lib.format.open_memmap(blend_path, mode="w+", dtype="float32",
                                            shape=(len(p_current),))
        for start in range(0, len(p_current), 100_000):
            stop = min(len(p_current), start + 100_000)
            blended[start:stop] = weight * p_current[start:stop] + (1-weight) * p_best[start:stop]
        blended.flush()
        blend_score, blend_threshold, _ = score_grid(dev_meta, blended, dev, cap)
        finalists.append({"trial": f"blend_current_{weight:.2f}_plus_{best_trial}",
                          "macro_f05": blend_score, "threshold": blend_threshold,
                          "sample_development_macro_f05": best_sample_blend["macro_f05"]})
        print(f"Full development blend weight={weight:.2f}: F0.5={blend_score:.6f}; "
              f"threshold={blend_threshold:.2f}", flush=True)
        del blended

    report = {
        "selection_split": "development",
        "final_validation_scored": False,
        "seed": SEED,
        "candidate_cap_per_source": cap,
        "development_entities_evaluated": eval_entities,
        "development_candidate_pairs_evaluated": len(dev_row_ix),
        "feature_names": list(FEATURES),
        "sampled_training_rows": len(sample_y),
        "sample_positive_rows": int(sample_y.sum()),
        "sample_negative_rows": int(len(sample_y) - sample_y.sum()),
        "trials": rows,
        "best_sample_trial": best_trial,
        "sample_development_ensembles": sample_blends,
        "best_sample_ensemble": best_sample_blend,
        "full_development_finalists": finalists,
        "best_sample_model": str(best_model_path),
    }
    out = tuning / "development_tuning_rank_uniform_advanced.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"Saved development-only tuning results: {out}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path,
                        default=root / "code/business_entity_resolution/artifacts/v2")
    parser.add_argument("--sample-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    run(args.artifacts_dir, args.sample_rows)


if __name__ == "__main__":
    main()
