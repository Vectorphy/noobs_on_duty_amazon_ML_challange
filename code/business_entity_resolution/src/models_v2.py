"""Optional tree-model adapters with fit checkpoints and progress logging."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import joblib
import numpy as np


TREE_COUNT = 300
MONITOR_ROWS = 20_000
MODEL_NAMES = ("lightgbm", "xgboost", "catboost")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def model_artifacts(root: Path, model_name: str) -> Path:
    return root if model_name == "lightgbm" else root / "models" / model_name


def make_model(model_name: str, memory_bounded: bool = False):
    threads = 1 if memory_bounded else (os.cpu_count() or 1)
    bins = 31 if memory_bounded else 63
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(n_estimators=TREE_COUNT, learning_rate=0.05,
                      num_leaves=15 if memory_bounded else 31,
                      min_child_samples=100, max_bin=bins, n_jobs=threads,
                      force_col_wise=memory_bounded, random_state=42,
                      importance_type="gain", verbosity=-1)
        return LGBMClassifier(**params), params
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install requirements-models.txt to use XGBoost") from exc
        params = dict(n_estimators=TREE_COUNT, learning_rate=0.05,
                      max_depth=8, min_child_weight=100, max_bin=bins,
                      tree_method="hist", n_jobs=threads, random_state=42,
                      eval_metric="logloss", verbosity=0)
        return XGBClassifier(**params), params
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install requirements-models.txt to use CatBoost") from exc
        params = dict(iterations=TREE_COUNT, learning_rate=0.05,
                      depth=8 if not memory_bounded else 6,
                      thread_count=threads, random_seed=42,
                      loss_function="Logloss", eval_metric="Logloss",
                      allow_writing_files=True, verbose=False)
        return CatBoostClassifier(**params), params
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class _LightGBMTensorBoard:
    order = 30
    before_iteration = False

    def __init__(self, writer):
        self.writer = writer

    def __call__(self, env) -> None:
        for dataset, metric, value, *_ in env.evaluation_result_list or ():
            self.writer.add_scalar(f"{dataset}/{metric}", float(value), env.iteration + 1)


def _xgboost_tensorboard(writer):
    if writer is None:
        return None
    import xgboost as xgb

    class Callback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            for dataset, metrics in evals_log.items():
                for metric, values in metrics.items():
                    value = values[-1]
                    if isinstance(value, tuple):
                        value = value[0]
                    writer.add_scalar(f"{dataset}/{metric}", float(value), epoch + 1)
            return False

    return Callback()


def _writer(tensorboard_dir: Path | None, stage: str):
    if tensorboard_dir is None:
        return None
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("TensorBoard is unavailable; install requirements-models.txt for event logs.", flush=True)
        return None
    return SummaryWriter(str(tensorboard_dir / safe_name(stage)))


def fit_model(model_name: str, model, x, y, *, stage: str,
              checkpoint_dir: Path, signature: dict,
              tensorboard_dir: Path | None = None):
    """Fit once per signature, reusing completed fits and native snapshots."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage = safe_name(stage)
    model_path = checkpoint_dir / f"{stage}.joblib"
    state_path = checkpoint_dir / f"{stage}.json"
    signature_text = json.dumps(signature, sort_keys=True, default=list)
    if model_path.exists() and state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("signature") == signature_text and saved.get("complete"):
            print(f"Reusing completed {model_name} fit checkpoint: {stage}", flush=True)
            return joblib.load(model_path)

    native_signature = checkpoint_dir / f"{stage}.signature.json"
    has_native = ((checkpoint_dir / f"{stage}.lgb.txt").exists()
                  or (checkpoint_dir / f"{stage}_xgb").exists()
                  or (checkpoint_dir / f"{stage}_catboost").exists())
    if has_native and (not native_signature.exists()
                       or native_signature.read_text(encoding="utf-8") != signature_text):
        native_path = checkpoint_dir / f"{stage}.lgb.txt"
        if native_path.exists():
            native_path.unlink()
        for folder in (checkpoint_dir / f"{stage}_xgb", checkpoint_dir / f"{stage}_catboost"):
            if folder.exists():
                shutil.rmtree(folder)
    native_signature.write_text(signature_text, encoding="utf-8")

    writer = _writer(tensorboard_dir, stage)
    try:
        # A fixed training-only subset supplies per-iteration loss monitoring.
        rng = np.random.default_rng(42)
        monitor_ix = rng.choice(len(y), min(len(y), MONITOR_ROWS), replace=False)
        eval_set = [(np.asarray(x[monitor_ix]), np.asarray(y[monitor_ix]))]
        started = time.monotonic()

        if model_name == "lightgbm":
            import lightgbm as lgb

            native_path = checkpoint_dir / f"{stage}.lgb.txt"

            def checkpoint(env):
                if (env.iteration + 1) % 50 == 0:
                    tmp = native_path.with_suffix(".tmp")
                    env.model.save_model(str(tmp))
                    os.replace(tmp, native_path)

            callbacks = [lgb.log_evaluation(period=25), checkpoint]
            if writer is not None:
                history = {}
                callbacks.extend((lgb.record_evaluation(history), _LightGBMTensorBoard(writer)))
            if native_path.exists():
                initial = lgb.Booster(model_file=str(native_path))
                remaining = max(1, TREE_COUNT - initial.current_iteration())
                model.set_params(n_estimators=remaining)
                model.fit(x, y, eval_set=eval_set, eval_names=["training_monitor"],
                          eval_metric="binary_logloss", callbacks=callbacks, init_model=initial)
            else:
                model.fit(x, y, eval_set=eval_set, eval_names=["training_monitor"],
                          eval_metric="binary_logloss", callbacks=callbacks)
            if writer is not None:
                writer.flush()
        elif model_name == "xgboost":
            import xgboost as xgb

            native_dir = checkpoint_dir / f"{stage}_xgb"
            native_dir.mkdir(exist_ok=True)
            callbacks = [xgb.callback.TrainingCheckPoint(
                directory=native_dir, name=stage, interval=50,
            )]
            tb_callback = _xgboost_tensorboard(writer)
            if tb_callback is not None:
                callbacks.append(tb_callback)
            model.set_params(callbacks=callbacks)
            snapshots = sorted(native_dir.glob(f"{stage}_*.ubj"),
                               key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
            resume = None
            if snapshots:
                resume = xgb.Booster()
                resume.load_model(str(snapshots[-1]))
                model.set_params(n_estimators=max(1, TREE_COUNT - resume.num_boosted_rounds()))
            model.fit(x, y, eval_set=eval_set, verbose=25, xgb_model=resume)
            # Callbacks may be local classes or hold a TensorBoard writer; do not
            # serialize those training-only objects into the reusable estimator.
            model.set_params(callbacks=None)
        else:
            native_dir = checkpoint_dir / f"{stage}_catboost"
            native_dir.mkdir(exist_ok=True)
            model.set_params(train_dir=str(native_dir))
            model.fit(x, y, eval_set=eval_set, verbose=25, save_snapshot=True,
                      use_best_model=False,
                      snapshot_file=str(native_dir / f"{stage}.cbsnapshot"),
                      snapshot_interval=60)

        tmp_model = model_path.with_suffix(".tmp")
        joblib.dump(model, tmp_model)
        os.replace(tmp_model, model_path)
        _atomic_json(state_path, {"signature": signature_text, "complete": True,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)})
        print(f"Saved {model_name} fit checkpoint: {model_path}", flush=True)
        return model
    finally:
        if writer is not None:
            writer.close()


def positive_probability(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.predict(x), dtype=np.float32)


def feature_importances(model) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    return np.asarray(model.get_feature_importance())
