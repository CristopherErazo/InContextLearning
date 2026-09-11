"""Evaluation of the trained model on a fixed, preprocessed test batch.

    batch, stats = preprocess_batch(generate_icl_batch(...), device)
    evaluator = Evaluator(scalars=[TopKAccuracy(1), LossMetric()],
                          artifacts=[ComposedMatrices()], loss_fn=loss_fn)
    metrics   = evaluator.scalars(model, batch, step)     # dict[str, float]
    artifacts = evaluator.artifacts(model, batch, step)   # {(name, group): (data, type)}

Both calls at the same `step` share one forward pass through `EvalContext`.
"""
from .batch import PreprocessStats, filter_batch, induction_mask, preprocess_batch
from .schedule import evaluation_steps, get_evaluation_times
from .evaluator import Artifacts, EvalContext, Evaluator, Probe, log_artifacts, split_on_off
from .scalars import LogitStatistics, LossMetric, TargetProbMass, TopKAccuracy
from .artifacts import AttentionMaps, ComposedMatrices, LogitHistograms, PerPositionOnOffLogits

__all__ = [
    # batch
    "preprocess_batch", "filter_batch", "induction_mask", "PreprocessStats",
    # schedule
    "evaluation_steps", "get_evaluation_times",
    # machinery
    "EvalContext", "Evaluator", "Probe", "Artifacts", "log_artifacts", "split_on_off",
    # scalar probes
    "LossMetric", "TopKAccuracy", "TargetProbMass", "LogitStatistics",
    # artifact probes
    "ComposedMatrices", "PerPositionOnOffLogits", "LogitHistograms", "AttentionMaps",
]
